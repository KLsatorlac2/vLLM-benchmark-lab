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

## Kaggle T4 ×2 / Qwen2.5-1.5B（2026-09-27）

环境：T4 16GB、`dtype=half`、**引擎 V0**（T4 不支持 V1）、XFormers、
`gpu_memory_utilization=0.85`。除 HF 臂外都是 V0。

**E1a 单请求对单请求**（10 token prompt，`max_tokens=64`）

| run | backend | batch 延迟 | output tok/s |
| --- | --- | --- | --- |
| `kaggle-1.5b-hf-single` | HF `generate`（逐条串行）| 2840.63 ms | 22.53 |
| `kaggle-1.5b-vllm-single` | vLLM，`max_num_seqs=1` | **1160.40 ms** | **55.15** |

→ **vLLM 延迟是 HF 的 0.408 倍（快 2.45×），吞吐 2.45 倍。**
本机 0.5B 同一步是 **3.7×**，但**这个差异归因不了**——
两台机器的 GPU、模型（0.5B vs 1.5B）、dtype（bf16 vs fp16）、引擎（V1 vs V0）
**四个变量同时不同**。唯一能说的是：**1.5B + T4 上，vLLM 的单请求优势仍有 2 倍以上。**

**E1b 批处理收益**（vLLM 自己 1 → 16 条）

| run | requests | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- |
| `kaggle-1.5b-vllm-single` | 1 | 1160.40 ms | 55.15 tok/s |
| `kaggle-1.5b-vllm-batch16` | 16 | 1212.59 ms | **844.47 tok/s** |

→ **吞吐 ×15.3，batch 延迟只涨 4.5%（×1.045）。**
本机是 **×16.6、延迟 ×0.965**。两台机器给出同一结论：
**1→16 区间里批处理几乎是把闲置算力直接换成吞吐。**
T4 上延迟开始有 4.5% 的可观测上升（本机是 −3.5%），说明 16 条已让 T4 略微吃紧。

**口径**：与 Part I 相同，都是离线 batch 数字（`latency_mode=batch`/`single`）。
HF 的 `tpot_ms=44.38` 是"总延迟 / 全部输出 token"的近似，不是排除首 token 后的 TPOT。
**Kaggle 侧的 server 模式实验失败**（Kaggle 不允许后台进程），
所以 T4 上**没有任何 per-request 数字**——详见报告 Part II 第 22 节。

分析见 `docs/benchmark_report.md` Part II 第 15 节。