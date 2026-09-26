"""Sample NVIDIA GPU metrics into timestamped JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


QUERY = "timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm"


def sample() -> list[dict[str, str]]:
    command = ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    fields = [field.split(",") for field in completed.stdout.strip().splitlines() if field.strip()]
    names = QUERY.split(",")
    return [dict(zip(names, [value.strip() for value in row])) for row in fields]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/raw/gpu_metrics.jsonl")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration
    with output.open("w", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            row = {"run_id": args.run_id, "sampled_at_unix": time.time(), "gpus": sample()}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            time.sleep(args.interval)
    print(f"wrote GPU metrics to {output}")


if __name__ == "__main__":
    main()
