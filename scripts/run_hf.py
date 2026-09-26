"""Run a small Transformers generation benchmark."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from common import environment_metadata, load_prompts, set_seed, write_jsonl


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=args.dtype,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    requests = load_prompts(args.prompts, args.requests, args.max_tokens)
    rows: list[dict[str, Any]] = []

    for request in requests:
        encoded = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=request.max_tokens,
                do_sample=False,
                return_dict_in_generate=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        input_tokens = encoded["input_ids"].shape[-1]
        output_tokens = output.sequences.shape[-1] - input_tokens
        rows.append(
            {
                "request_id": request.request_id,
                "backend": "hf",
                "status": "ok",
                "input_tokens": input_tokens,
                "output_tokens": int(output_tokens),
                "latency_ms": elapsed_ms,
                "ttft_ms": None,
                "tpot_ms": elapsed_ms / max(int(output_tokens), 1),
                "output_tok_per_s": output_tokens / max(elapsed_ms / 1000, 1e-9),
                "prompt": request.prompt if args.include_prompts else None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompts")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/raw/hf.jsonl")
    parser.add_argument("--include-prompts", action="store_true")
    args = parser.parse_args()
    if args.dtype == "auto":
        args.dtype = "auto"
    else:
        args.dtype = getattr(__import__("torch"), args.dtype)
    rows = run(args)
    metadata = {"runner": "run_hf", "environment": environment_metadata(), "args": vars(args)}
    write_jsonl(args.output, [{"type": "metadata", **metadata}, *rows])
    print(f"wrote {len(rows)} requests to {Path(args.output)}")


if __name__ == "__main__":
    main()
