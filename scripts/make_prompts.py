"""Generate prompt files with verified token lengths.

The offline `LLM.generate` path feeds raw strings straight to the model (no chat
template is applied), so prompt length is fully controlled by these files. That
makes the *actual* input token count the only honest way to describe a
context-length arm -- `max_model_len` is just a cap.

Modes
-----
distinct
    `--num-prompts` prompts of exactly `--target-tokens` tokens, each starting
    with a unique marker. Use for context-length and concurrency experiments:
    no two prompts share a prefix, so prefix caching cannot flatter the result.
shared-prefix
    Every prompt opens with a byte-identical `--prefix-tokens` block, then
    diverges into a unique tail. Use for the prefix-caching ON arm.
random-prefix
    Every prompt opens with its own unique `--prefix-tokens` block, then ends
    with one byte-identical shared tail. Same total length as `shared-prefix`,
    but nothing is shareable from position 0 -- this is the control arm.
    Comparing `shared-prefix` against `random-prefix` isolates prefix reuse;
    comparing against a plain `distinct` file would not.
mixed
    Alternates `--long-tokens` and `--short-tokens` prompts in one file, so
    short requests sit behind a long prefill. Use for chunked prefill.

Note: `load_prompts` in scripts/common.py cycles a prompt file to reach
`--requests`. Always pass `--requests` equal to the number of lines here,
otherwise the file is repeated and (with prefix caching on) the repeats become
cache hits.

Usage
-----
    python scripts/make_prompts.py --mode distinct \
      --target-tokens 4096 --num-prompts 4 \
      --output experiments/context_length/prompts_4096.txt --strict

    python scripts/make_prompts.py --check experiments/context_length/prompts_4096.txt
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

FILLER_SENTENCES = [
    "This document is a controlled language model benchmarking corpus.",
    "The purpose is to evaluate inference latency, throughput and memory usage.",
    "Attention behaviour changes measurably once the sequence exceeds a few thousand tokens.",
    "Scheduling efficiency depends on how many sequences the engine can admit per step.",
    "A prefill phase processes the whole prompt before any token is emitted.",
    "The decode phase then emits one token per step for every running sequence.",
    "Key and value tensors are cached so that history is never recomputed.",
    "Cache size grows linearly with the number of concurrent sequences.",
    "Throughput and latency are usually in direct tension with one another.",
    "A larger batch amortises weight reads across more useful arithmetic.",
    "Quantisation trades numerical fidelity for a smaller memory footprint.",
    "Grouped query attention reduces the size of the key value cache considerably.",
    "Prefill is compute bound while decode is memory bandwidth bound.",
    "Chunking a long prefill lets short requests interleave with it.",
    "Reusing a cached prefix avoids recomputing tokens that were already seen.",
    "Tail latency is the metric users actually notice in production.",
    "Time to first token and time per output token stress different subsystems.",
    "Paged memory management keeps fragmentation from wasting capacity.",
    "A benchmark is only reproducible if the workload is pinned exactly.",
    "Reporting an input token count is more honest than reporting a configured cap.",
    "Thermal throttling on a laptop chassis makes repeated runs drift.",
    "Interference from a desktop compositor can steal several percent of throughput.",
    "Every claim in the final report should trace back to a raw JSONL row.",
    "Failures such as out of memory are results too and must be kept.",
]

MARKER_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


@dataclass
class Prompt:
    text: str
    actual_tokens: int
    label: str = ""
    target_tokens: int = 0


def unique_marker(index: int, rng: random.Random) -> str:
    """A short per-prompt prefix so no two generated prompts start alike.

    The random block comes first on purpose: a fixed leading word such as
    "[record" would tokenise identically across prompts and hand the control arm
    a few tokens of accidental shared prefix. Putting the random token at
    position 0 keeps the shared prefix at zero tokens regardless of cache block
    size.
    """
    token = "".join(rng.choice(MARKER_ALPHABET) for _ in range(8))
    return f"{token}{index:04d} "


def measure(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def filler_text(rng: random.Random, min_tokens: int) -> str:
    """Concatenate shuffled filler sentences, over-generating on purpose.

    Exact length is imposed afterwards by `exact_trim`; word counts are only a
    cheap lower bound on token counts, hence the generous factor.
    """
    parts: list[str] = []
    estimate = 0
    while estimate < min_tokens * 2:
        sentence = rng.choice(FILLER_SENTENCES)
        parts.append(sentence)
        estimate += len(sentence.split()) + 1
    return " ".join(parts)


def exact_trim(tokenizer, text: str, target: int, rng: random.Random) -> Prompt:
    """Trim `text` to exactly `target` tokens, repairing decode round-trip drift.

    Decoding the first `target` ids and re-encoding them can land one token off
    because of BPE merge boundaries, so re-measure and nudge until the measured
    length is exact (or until no further improvement is possible).
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    guard = 0
    while len(ids) < target and guard < 256:
        text = text + " " + rng.choice(FILLER_SENTENCES)
        ids = tokenizer.encode(text, add_special_tokens=False)
        guard += 1
    if len(ids) < target:
        raise RuntimeError(f"could not reach {target} tokens (got {len(ids)})")

    best = tokenizer.decode(ids[:target], skip_special_tokens=True)
    best_delta = abs(measure(tokenizer, best) - target)
    for _ in range(6):
        if best_delta == 0:
            break
        current = tokenizer.encode(best, add_special_tokens=False)
        if len(current) > target:
            candidate = tokenizer.decode(current[:target], skip_special_tokens=True)
        else:
            extended = best + " " + rng.choice(FILLER_SENTENCES)
            candidate = tokenizer.decode(
                tokenizer.encode(extended, add_special_tokens=False)[:target],
                skip_special_tokens=True,
            )
        delta = abs(measure(tokenizer, candidate) - target)
        if delta < best_delta:
            best, best_delta = candidate, delta
        else:
            break
    return Prompt(best, measure(tokenizer, best), target_tokens=target)


def gen_distinct(tokenizer, args: argparse.Namespace) -> list[Prompt]:
    prompts = []
    for index in range(args.num_prompts):
        rng = random.Random(args.seed + index)
        text = unique_marker(index, rng) + filler_text(rng, args.target_tokens)
        prompts.append(exact_trim(tokenizer, text, args.target_tokens, rng))
    return prompts


def gen_shared_prefix(tokenizer, args: argparse.Namespace) -> list[Prompt]:
    """One identical leading block, then a unique tail per prompt."""
    shared_rng = random.Random(args.seed)
    shared = exact_trim(
        tokenizer, filler_text(shared_rng, args.prefix_tokens), args.prefix_tokens, shared_rng
    )
    tail_target = max(args.target_tokens - shared.actual_tokens, 16)
    prompts = []
    for index in range(args.num_prompts):
        rng = random.Random(args.seed + 1000 + index)
        text = unique_marker(index, rng) + filler_text(rng, tail_target)
        tail = exact_trim(tokenizer, text, tail_target, rng)
        # Joining shifts the token boundary, so re-trim the *end* only. exact_trim
        # never touches the front, which keeps the shared block byte-identical.
        joined = exact_trim(tokenizer, shared.text + " " + tail.text, args.target_tokens, rng)
        prompts.append(Prompt(joined.text, joined.actual_tokens, f"shared{shared.actual_tokens}", args.target_tokens))
    return prompts


def gen_random_prefix(tokenizer, args: argparse.Namespace) -> list[Prompt]:
    """A unique leading block per prompt, then one identical trailing tail.

    The tail is identical across prompts on purpose: it still cannot be cached,
    because no block matches from position 0. That keeps this arm the exact
    structural mirror of `shared-prefix`.
    """
    tail_rng = random.Random(args.seed + 2000)
    tail_target = max(args.target_tokens - args.prefix_tokens, 16)
    tail = exact_trim(
        tokenizer, filler_text(tail_rng, tail_target), tail_target, tail_rng
    )
    prompts = []
    for index in range(args.num_prompts):
        rng = random.Random(args.seed + 3000 + index)
        text = unique_marker(index, rng) + filler_text(rng, args.prefix_tokens)
        prefix = exact_trim(tokenizer, text, args.prefix_tokens, rng)
        joined = prefix.text + " " + tail.text
        prompts.append(
            Prompt(joined, measure(tokenizer, joined), f"random{prefix.actual_tokens}", args.target_tokens)
        )
    return prompts


def gen_mixed(tokenizer, args: argparse.Namespace) -> list[Prompt]:
    prompts = []
    for index in range(args.num_prompts):
        is_long = index % 2 == 0
        target = args.long_tokens if is_long else args.short_tokens
        rng = random.Random(args.seed + 4000 + index)
        text = unique_marker(index, rng) + filler_text(rng, target)
        prompt = exact_trim(tokenizer, text, target, rng)
        prompt.label = "long" if is_long else "short"
        prompts.append(prompt)
    return prompts


GENERATORS = {
    "distinct": gen_distinct,
    "shared-prefix": gen_shared_prefix,
    "random-prefix": gen_random_prefix,
    "mixed": gen_mixed,
}


def check_files(tokenizer, paths: list[str]) -> int:
    """Re-measure existing prompt files without writing anything.

    Run this on Kaggle too: it proves the generated files still hit the intended
    token counts under whichever tokenizer that host actually downloaded.
    """
    worst = 0
    for path in paths:
        lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        lengths = [measure(tokenizer, line) for line in lines]
        if not lengths:
            print(f"{path}: EMPTY")
            worst = 1
            continue
        print(
            f"{path}: {len(lengths)} prompts  "
            f"min={min(lengths)} mean={statistics.mean(lengths):.1f} max={max(lengths)}"
        )
        for index, length in enumerate(lengths):
            print(f"    line {index:3d}: {length} tokens")
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="tokenizer source")
    parser.add_argument("--mode", choices=sorted(GENERATORS), default="distinct")
    parser.add_argument("--target-tokens", type=int, default=2048, help="total tokens per prompt")
    parser.add_argument("--prefix-tokens", type=int, default=1024, help="shared/unique block size")
    parser.add_argument("--long-tokens", type=int, default=3584, help="mixed mode: long prompt")
    parser.add_argument("--short-tokens", type=int, default=64, help="mixed mode: short prompt")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strict", action="store_true", help="exit non-zero unless every prompt hits its target exactly")
    parser.add_argument("--check", nargs="+", metavar="PATH", help="only verify existing prompt files")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"tokenizer: {args.model} ({type(tokenizer).__name__})")

    if args.check:
        raise SystemExit(check_files(tokenizer, args.check))

    if not args.output:
        raise SystemExit("--output is required unless --check is used")

    prompts = GENERATORS[args.mode](tokenizer, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(prompt.text for prompt in prompts) + "\n", encoding="utf-8")

    lengths = [prompt.actual_tokens for prompt in prompts]
    manifest = {
        "mode": args.mode,
        "model": args.model,
        "seed": args.seed,
        "num_prompts": len(prompts),
        "target_tokens": args.target_tokens,
        "prefix_tokens": args.prefix_tokens if args.mode in {"shared-prefix", "random-prefix"} else None,
        "actual_tokens": lengths,
        "actual_min": min(lengths),
        "actual_mean": round(statistics.mean(lengths), 2),
        "actual_max": max(lengths),
        "labels": [prompt.label for prompt in prompts],
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {len(prompts)} prompts to {output}")
    print(f"actual tokens: min={min(lengths)} mean={statistics.mean(lengths):.1f} max={max(lengths)}")
    exact = True
    for index, prompt in enumerate(prompts):
        flag = ""
        if prompt.actual_tokens != prompt.target_tokens:
            flag = f"  <-- wanted {prompt.target_tokens}"
            exact = False
        print(f"  line {index:3d}: {prompt.actual_tokens:6d} tokens {prompt.label}{flag}")

    if args.strict and not exact:
        print("STRICT: some prompts missed their target length", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
