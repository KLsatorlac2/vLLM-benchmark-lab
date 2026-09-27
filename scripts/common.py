"""Shared workload, timing, and JSONL helpers for the benchmark lab."""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RequestSpec:
    request_id: str
    prompt: str
    input_tokens_target: int | None = None
    max_tokens: int = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def load_prompts(path: str | None, count: int, max_tokens: int) -> list[RequestSpec]:
    if path:
        prompts = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    else:
        prompts = [
            "Explain why batching improves language model inference throughput.",
            "Compare time to first token and time per output token.",
            "Describe one tradeoff of increasing the context window.",
            "What does KV cache reuse optimize in autoregressive decoding?",
        ]
    prompts = (prompts * ((count + len(prompts) - 1) // len(prompts)))[:count]
    return [RequestSpec(str(uuid.uuid4()), prompt, None, max_tokens) for prompt in prompts]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def now_ns() -> int:
    return time.perf_counter_ns()


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile, so it never invents a value between samples."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def ns_to_ms(value: int) -> float:
    return value / 1_000_000


# Environment variables that change what the numbers mean. `VLLM_USE_V1` picks the
# engine (V0 vs V1 differ by ~10% on identical configs, see docs/runbook_three_models.md
# pitfall 7), the attention backend decides how long context performs, and
# `CUDA_VISIBLE_DEVICES` decides which GPUs a tensor-parallel run actually used.
# Recording them is the difference between a result that can be attributed and one
# that cannot; an unset variable is recorded explicitly rather than omitted.
TRACKED_ENV_VARS = (
    "VLLM_USE_V1",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_ATTENTION_BACKEND",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
)


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "python": os.sys.version.split()[0],
        "env_vars": {name: os.environ.get(name, "<unset>") for name in TRACKED_ENV_VARS},
    }
    try:
        import transformers

        metadata["transformers"] = transformers.__version__
    except Exception as exc:  # pragma: no cover - optional dependency
        metadata["transformers_error"] = str(exc)
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        metadata["torch_error"] = str(exc)
    try:
        import vllm

        metadata["vllm"] = vllm.__version__
    except Exception as exc:  # pragma: no cover - optional dependency
        metadata["vllm_error"] = str(exc)
    return metadata


def request_as_dict(request: RequestSpec) -> dict[str, Any]:
    return asdict(request)
