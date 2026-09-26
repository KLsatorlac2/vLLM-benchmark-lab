"""Run offline vLLM generation with a reproducible JSONL output."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from common import environment_metadata, load_prompts, set_seed, write_jsonl


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    set_seed(args.seed)
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        # Passed explicitly, including the False case: vLLM V1 turns prefix
        # caching on by default, so an omitted flag would silently mean "on".
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    requests = load_prompts(args.prompts, args.requests, args.max_tokens)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [request.prompt for request in requests]
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed_ms = (time.perf_counter() - start) * 1000
    rows: list[dict[str, Any]] = []
    for request, output in zip(requests, outputs):
        completion = output.outputs[0]
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(completion.token_ids)
        rows.append(
            {
                "request_id": request.request_id,
                "backend": "vllm_offline",
                "status": "ok",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": elapsed_ms,
                "ttft_ms": None,
                "tpot_ms": None,
                "output_tok_per_s": sum(len(item.outputs[0].token_ids) for item in outputs)
                / max(elapsed_ms / 1000, 1e-9),
                "prompt": request.prompt if args.include_prompts else None,
                "text": completion.text if args.include_text else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="token budget per scheduler step; pin it for chunked-prefill A/B runs",
    )
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--prompts")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/raw/vllm.jsonl")
    parser.add_argument("--include-prompts", action="store_true")
    parser.add_argument("--include-text", action="store_true")
    args = parser.parse_args()
    rows = run(args)
    metadata = {"runner": "run_vllm", "environment": environment_metadata(), "args": vars(args)}
    write_jsonl(args.output, [{"type": "metadata", **metadata}, *rows])
    print(f"wrote {len(rows)} requests to {Path(args.output)}")


if __name__ == "__main__":
    main()
