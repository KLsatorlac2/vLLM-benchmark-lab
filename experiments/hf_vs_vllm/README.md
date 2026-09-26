# HF vs vLLM

固定模型、prompt、输出上限和随机种子，分别运行 `run_hf.py` 与 `run_vllm.py`。本机验收使用 Qwen2.5-0.5B；Kaggle 再扩展 1.5B/7B。比较 latency、TTFT/TPOT 可用性、output tok/s、显存和输出 token 数。

## 本轮结果（2026-09-26，Qwen2.5-0.5B，RTX 4060）

拆成两步，把"引擎差异"和"批处理差异"分开：

**E1a 单请求对单请求**（10 token prompt，`max_tokens=64`）

| run | backend | batch 延迟 | output tok/s |
| --- | --- | --- | --- |
| `local-0.5b-hf-single` | HF `generate`（逐条串行）| 3675.41 ms | 17.41 |
| `local-0.5b-vllm-single` | vLLM，`max_num_seqs=1` | **986.91 ms** | **64.85** |

→ **vLLM 延迟是 HF 的 0.269 倍（快 3.7×），吞吐 3.72 倍。** 两边都是 batch=1，差异纯来自引擎。

**E1b 批处理收益**（vLLM 自己 1 → 16 条）

| run | requests | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- |
| `local-0.5b-vllm-single` | 1 | 986.91 ms | 64.85 tok/s |
| `local-0.5b-vllm-batch16` | 16 | **952.11 ms** | **1075.51 tok/s** |

→ **吞吐 ×16.6，而 batch 延迟几乎不变（×0.965）。** 单请求时 GPU 远未饱和。

**口径**：这两组都是离线 batch 数字（`latency_mode=batch`/`single`）。
HF 的 `tpot_ms=57.43` 是"总延迟 / 全部输出 token"的近似，不是排除首 token 后的 TPOT。
需要真实 TTFT/TPOT/p95 时走 server 模式（见 `docs/runbook_three_models.md` 4.7 节）。

分析见 `docs/benchmark_report.md` 第 5 节。