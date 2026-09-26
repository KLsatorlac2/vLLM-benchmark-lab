"""Per-request TTFT / TPOT measurement against an OpenAI-compatible vLLM server.

The offline `LLM.generate` runners measure one whole batch: every row in their
JSONL carries the same latency, so p95 and TTFT are simply unavailable there.
This script drives the HTTP server instead, which is the only path that can
report per-request queueing, TTFT and TPOT.

It deliberately calls the legacy `/v1/completions` endpoint with a raw prompt
string. That matches what `LLM.generate` does offline (no chat template), so the
two result sets stay comparable. Use `/v1/chat/completions` only if you are
explicitly benchmarking the chat template.

Load model: closed-loop. `--concurrency` requests are kept in flight at a time
until `--requests` have been served. `--concurrency` equal to `--requests`
reproduces the offline burst; a smaller value measures queueing just like
lowering `max_num_seqs` does.

Start the server first, e.g.:

    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \\
      --model Qwen/Qwen2.5-0.5B-Instruct \\
      --host 127.0.0.1 --port 8000 \\
      --max-model-len 4096 --max-num-seqs 16 \\
      --served-model-name qwen0.5b \\
      --no-enable-prefix-caching

Then:

    python scripts/benchmark_http.py \\
      --model qwen0.5b --prompts experiments/concurrency/prompts_512x64.txt \\
      --requests 64 --concurrency 16 --max-tokens 64 \\
      --output results/raw/local-0.5b-http-concurrency-16.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from common import environment_metadata, load_prompts, percentile, write_jsonl

# vLLM emits one chunk per output token while streaming, so counting chunks is a
# good token count as long as stream_options is unavailable. When the server does
# report usage we prefer that number instead. Flipped off after the first
# rejection so we only pay for the failed attempt once.
USAGE_SUPPORTED = True
USAGE_LOCK = threading.Lock()


def consume_stream(stream, start: float) -> tuple[float | None, int, int | None, int | None]:
    """Walk the stream as it arrives, timestamping the first content chunk.

    Returns (ttft_ms, chunks_with_text, usage_completion_tokens, usage_prompt_tokens).
    """
    ttft_ms = None
    produced = 0
    usage_tokens = None
    input_tokens = None
    for chunk in stream:
        now = time.perf_counter()
        text = chunk.choices[0].text if chunk.choices else None
        if text:
            if ttft_ms is None:
                ttft_ms = (now - start) * 1000
            produced += 1
        usage = getattr(chunk, "usage", None)
        if usage:
            usage_tokens = usage.completion_tokens
            input_tokens = usage.prompt_tokens
    return ttft_ms, produced, usage_tokens, input_tokens


def one_request(client, args: argparse.Namespace, prompt: str, request_id: str) -> dict[str, Any]:
    global USAGE_SUPPORTED
    row: dict[str, Any] = {
        "request_id": request_id,
        "backend": "vllm_server",
        "server_mode": True,
        "status": "error",
        "input_tokens": None,
        "output_tokens": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "latency_ms": None,
        "output_tok_per_s": None,
        "error": None,
    }
    start = time.perf_counter()
    try:
        with USAGE_LOCK:
            use_usage = USAGE_SUPPORTED
        kwargs: dict[str, Any] = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        if use_usage:
            kwargs["stream_options"] = {"include_usage": True}
        try:
            # The stream must be consumed lazily: timestamping chunks after
            # materialising them into a list would stamp every token with the
            # time the *last* one arrived, collapsing TTFT into latency.
            stream = client.completions.create(**kwargs)
            ttft_ms, produced, usage_tokens, input_tokens = consume_stream(stream, start)
        except Exception:
            if not use_usage:
                raise
            with USAGE_LOCK:
                USAGE_SUPPORTED = False
            kwargs.pop("stream_options", None)
            stream = client.completions.create(**kwargs)
            ttft_ms, produced, usage_tokens, input_tokens = consume_stream(stream, start)

        end = time.perf_counter()
        latency_ms = (end - start) * 1000
        output_tokens = usage_tokens if usage_tokens is not None else produced
        decode_ms = max(latency_ms - (ttft_ms if ttft_ms is not None else 0.0), 1e-9)
        row.update(
            {
                "status": "ok",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": ttft_ms,
                "tpot_ms": decode_ms / max(output_tokens - 1, 1),
                "latency_ms": latency_ms,
                "output_tok_per_s": output_tokens / max(latency_ms / 1000, 1e-9),
            }
        )
    except Exception as exc:  # keep the failure as data, never swallow it silently
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_ms"] = (time.perf_counter() - start) * 1000
    return row


def summarize(rows: list[dict[str, Any]], wall_ms: float, concurrency: int) -> dict[str, Any]:
    ok = [row for row in rows if row["status"] == "ok"]
    ttft = [row["ttft_ms"] for row in ok if row["ttft_ms"] is not None]
    latency = [row["latency_ms"] for row in ok if row["latency_ms"] is not None]
    tpot = [row["tpot_ms"] for row in ok if row["tpot_ms"] is not None]
    output_tokens = sum(row["output_tokens"] or 0 for row in ok)
    return {
        "type": "summary",
        "backend": "vllm_server",
        "server_mode": True,
        "status": "ok" if ok else "failed",
        "requests": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "concurrency": concurrency,
        "wall_ms": wall_ms,
        "output_tokens": output_tokens,
        "req_per_s": len(ok) / max(wall_ms / 1000, 1e-9),
        "output_tok_per_s": output_tokens / max(wall_ms / 1000, 1e-9),
        "mean_ttft_ms": statistics.mean(ttft) if ttft else None,
        "p50_ttft_ms": percentile(ttft, 0.50),
        "p95_ttft_ms": percentile(ttft, 0.95),
        "p99_ttft_ms": percentile(ttft, 0.99),
        "mean_tpot_ms": statistics.mean(tpot) if tpot else None,
        "p95_tpot_ms": percentile(tpot, 0.95),
        "mean_latency_ms": statistics.mean(latency) if latency else None,
        "p50_latency_ms": percentile(latency, 0.50),
        "p95_latency_ms": percentile(latency, 0.95),
        "p99_latency_ms": percentile(latency, 0.99),
        "errors": [row["error"] for row in rows if row["error"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="served model name (must match --served-model-name)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--prompts")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=16, help="requests kept in flight at once")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2, help="requests sent and discarded before measuring")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", default="results/raw/http.jsonl")
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    requests = load_prompts(args.prompts, args.requests + args.warmup, args.max_tokens)

    for index, request in enumerate(requests[: args.warmup]):
        one_request(client, args, request.prompt, f"warmup-{index}")
    print(f"warmup done ({args.warmup} requests discarded)")

    measured = requests[args.warmup :]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as pool:
        futures = [
            pool.submit(one_request, client, args, request.prompt, request.request_id)
            for request in measured
        ]
        rows = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - start) * 1000

    summary = summarize(rows, wall_ms, args.concurrency)
    metadata = {
        "type": "metadata",
        "runner": "benchmark_http",
        "server_mode": True,
        "environment": environment_metadata(),
        "args": vars(args),
    }
    write_jsonl(args.output, [metadata, *rows, summary])
    print(json.dumps({key: value for key, value in summary.items() if key != "errors"}, indent=2))
    if summary["errors"]:
        print(f"first error: {summary['errors'][0]}")
    if not any(row["input_tokens"] for row in rows if row["status"] == "ok"):
        print(
            "WARNING: the server did not report usage, so input_tokens is null and "
            "output_tokens counts streamed chunks (one per token on vLLM). Use the "
            "offline runner's input_tokens, or `make_prompts.py --check`, for the "
            "authoritative prompt length."
        )
    print(f"wrote {len(rows)} requests to {Path(args.output)}")


if __name__ == "__main__":
    main()
