"""Aggregate raw JSONL runs into report-ready tables.

Reads the files written by run_hf.py / run_vllm.py / benchmark_http.py and prints
a markdown table plus a GPU table, so filling in docs/benchmark_report.md is a
copy-paste rather than a manual pass over raw rows.

It also labels each run's latency semantics. Offline `LLM.generate` rows all
carry the same batch latency, so a p95 computed over them is meaningless; those
runs are marked `batch` and their percentiles are suppressed. Server runs and HF
runs are marked `per-request` and keep their percentiles.

Usage
-----
    python scripts/summarize_results.py results/raw/local-0.5b-*.jsonl
    python scripts/summarize_results.py results/raw/kaggle-1.5b-*.jsonl \
      --gpu results/raw/gpu-kaggle-1.5b.jsonl \
      --output-md results/processed/summary-kaggle-1.5b.md \
      --output-csv results/processed/summary-kaggle-1.5b.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from common import percentile, read_jsonl

COLUMNS = [
    "run",
    "model",
    "backend",
    "requests",
    "in_tokens",
    "out_tokens",
    "latency_mode",
    "latency_ms",
    "p95_ms",
    "ttft_ms",
    "tpot_ms",
    "out_tok_per_s",
    "status",
]


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_run(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    metadata = next((row for row in rows if row.get("type") == "metadata"), {})
    summary = next((row for row in rows if row.get("type") == "summary"), None)
    samples = [
        row
        for row in rows
        if row.get("type") not in {"metadata", "summary"} and row.get("status") == "ok"
    ]
    failures = [row for row in rows if row.get("status") == "error"]
    args = metadata.get("args", {}) or {}

    latency = [as_float(row.get("latency_ms")) for row in samples]
    latency = [value for value in latency if value is not None]
    ttft = [as_float(row.get("ttft_ms")) for row in samples]
    ttft = [value for value in ttft if value is not None]
    tpot = [as_float(row.get("tpot_ms")) for row in samples]
    tpot = [value for value in tpot if value is not None]
    input_tokens = [int(row.get("input_tokens") or 0) for row in samples]
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in samples)

    # One distinct latency value across several requests means the runner timed a
    # whole batch, not individual requests.
    mode = "batch" if len(latency) > 1 and len({round(value, 6) for value in latency}) == 1 else "per-request"
    if len(latency) <= 1:
        mode = "single"

    if summary:
        throughput = as_float(summary.get("output_tok_per_s"))
        p95 = as_float(summary.get("p95_latency_ms"))
        ttft_value = as_float(summary.get("mean_ttft_ms"))
        tpot_value = as_float(summary.get("mean_tpot_ms"))
    else:
        throughput = None
        if samples:
            per_request = [as_float(row.get("output_tok_per_s")) for row in samples]
            per_request = [value for value in per_request if value is not None]
            throughput = statistics.mean(per_request) if per_request else None
        p95 = percentile(latency, 0.95) if mode == "per-request" else None
        ttft_value = statistics.mean(ttft) if ttft else None
        tpot_value = statistics.mean(tpot) if tpot else None

    return {
        "run": path.name,
        "model": args.get("model", "?"),
        "backend": (samples[0].get("backend") if samples else metadata.get("runner", "?")),
        "requests": len(samples),
        "in_tokens": (
            f"{statistics.mean(input_tokens):.0f}({max(input_tokens)})" if input_tokens else "?"
        ),
        "out_tokens": output_tokens,
        "latency_mode": mode,
        "latency_ms": round(statistics.mean(latency), 2) if latency else None,
        "p95_ms": round(p95, 2) if p95 is not None else None,
        "ttft_ms": round(ttft_value, 2) if ttft_value is not None else None,
        "tpot_ms": round(tpot_value, 2) if tpot_value is not None else None,
        "out_tok_per_s": round(throughput, 2) if throughput is not None else None,
        "status": "ok" if samples else "failed",
        "failures": len(failures),
        "input_tokens_mean": round(statistics.mean(input_tokens), 1) if input_tokens else None,
        "input_tokens_max": max(input_tokens) if input_tokens else None,
        "batch_latency_ms": round(latency[0], 2) if mode == "batch" and latency else None,
    }


def summarize_gpu(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    per_gpu: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        for gpu in row.get("gpus", []):
            key = (str(gpu.get("index")), str(gpu.get("name")))
            bucket = per_gpu.setdefault(
                key, {"memory.used": [], "memory.total": [], "utilization.gpu": [], "power.draw": [], "temperature.gpu": []}
            )
            for field in bucket:
                value = as_float(gpu.get(field))
                if value is not None:
                    bucket[field].append(value)

    out = []
    for (index, name), bucket in sorted(per_gpu.items()):
        used = bucket["memory.used"]
        total = bucket["memory.total"]
        out.append(
            {
                "gpu": index,
                "name": name,
                "samples": len(used),
                "peak_mem_used_mb": round(max(used), 1) if used else None,
                "mem_total_mb": round(max(total), 1) if total else None,
                "peak_mem_pct": round(max(used) / max(total) * 100, 1) if used and total else None,
                "mean_util_pct": round(statistics.mean(bucket["utilization.gpu"]), 1) if bucket["utilization.gpu"] else None,
                "max_util_pct": round(max(bucket["utilization.gpu"]), 1) if bucket["utilization.gpu"] else None,
                "max_power_w": round(max(bucket["power.draw"]), 1) if bucket["power.draw"] else None,
                "mean_temp_c": round(statistics.mean(bucket["temperature.gpu"]), 1) if bucket["temperature.gpu"] else None,
            }
        )
    return out


def cell(value: Any) -> str:
    return "n/a" if value is None else str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="raw JSONL files from results/raw/")
    parser.add_argument("--gpu", default=None, help="GPU metrics JSONL collected alongside the runs")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    runs = [load_run(Path(path)) for path in args.inputs]
    table = markdown_table(runs, COLUMNS)
    print("\n## Runs\n")
    print(table)

    batch_runs = [row["run"] for row in runs if row["latency_mode"] == "batch"]
    if batch_runs:
        print(
            "\n`latency_mode=batch` means every request row shares one measurement of the whole "
            "batch: `latency_ms` is batch latency, `out_tok_per_s` is batch throughput, and p95 "
            "TTFT/TPOT are not available. See project_summary.md section 6.\n"
            f"Affected runs: {', '.join(batch_runs)}"
        )
    failed = [row for row in runs if row["status"] == "failed" or row["failures"]]
    if failed:
        print("\nRuns with failures (keep these as results, do not delete):")
        for row in failed:
            print(f"  {row['run']}: status={row['status']} failed_rows={row['failures']}")

    gpu_rows = []
    if args.gpu:
        gpu_rows = summarize_gpu(Path(args.gpu))
        if gpu_rows:
            print("\n## GPU\n")
            print(markdown_table(gpu_rows, list(gpu_rows[0].keys())))

    if args.output_md:
        body = ["# Benchmark summary", "", "## Runs", "", table]
        if batch_runs:
            body += ["", f"Batch-timed runs (batch latency / batch throughput only): {', '.join(batch_runs)}"]
        if gpu_rows:
            body += ["", "## GPU", "", markdown_table(gpu_rows, list(gpu_rows[0].keys()))]
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text("\n".join(body) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output_md}")

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(runs[0].keys()))
            writer.writeheader()
            writer.writerows(runs)
        print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
