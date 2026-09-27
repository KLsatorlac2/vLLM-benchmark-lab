# vLLM Benchmark Lab Project Summary

## 1. Project Overview

项目名称：`vllm-benchmark-lab`

项目目标：把 LLM 推理优化从主观感受转化为可复现实验数据，比较 HuggingFace Transformers 与 vLLM 在不同模型、并发、上下文长度、缓存策略和量化策略下的性能差异。

实验矩阵（E1–E7）与三模型分工的完整定义见 `docs/runbook_three_models.md` 第 2 节。

当前进度（**2026-09-27**，三模型矩阵完成 2/3）：

**本机 RTX 4060 / Qwen2.5-0.5B（2026-09-26）**

1. HF vs vLLM：0.5B — **已完成**（单请求 + 批处理两步拆开）
2. 并发：`max_num_seqs` = 16、32、64 — **已完成**（128/256 未扫，OOM 边界未知）
3. 上下文长度：512、2048、4096、8192、16384 — **已完成**
4. Prefix caching：开启/关闭 × 共享前缀/随机前缀 — **已完成**（四臂对照）
5. Chunked prefill：开启/关闭（V0）+ V1 参考臂 — **已完成**，但只有离线 batch 数字
6. 量化：AWQ、fp8 KV cache — **已完成**；GPTQ **未执行**
7. 并行：TP=1 vs TP=2 — **跳过**（本机单卡）

**Kaggle T4 ×2 / Qwen2.5-1.5B（2026-09-27）**

8. E1–E7 全矩阵 — **已完成**，22 个产物文件（20 个 vLLM run + 1 个 HF run + 1 个失败的 server run）
   - E2 并发扫到 **16/64/128/256**（本机没扫到的两档，在 T4 上补上了）
   - E6 量化只有 **AWQ**（GPTQ 未执行；fp8 KV **T4 不支持**）
   - E7 **TP=1 vs TP=2 已完成**——实测 **TP=2 快 24.8%**，推翻了 runbook 的预设
9. E8 HTTP server — **失败**（Kaggle 不允许后台进程，64/64 `APIConnectionError`），
   **保留为失败实验**，见 `results/raw/kaggle-1.5b-http-concurrency-16.jsonl`

**未执行**

10. Kaggle 7B（AWQ / GPTQ）— **未执行**
11. E7b 两个独立单卡实例 — **未执行**

结论见 `docs/benchmark_report.md`（**Part I** 第 1–12 节 = 本机 0.5B，**Part II** 第 13–23 节 = Kaggle 1.5B）。
两部分**绝对数字不可横向比较**（GPU / 模型 / dtype / 引擎四个变量同时不同），见报告第 23.1 节。

聚合表（`results/processed/summary-*.md`）**不进版本库**，本地用
`scripts/summarize_results.py` 重建，见 README 说明。

## 2. Hardware And Runtime

本机环境：

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 显存：8GB
- GPU 数量：1
- Python：3.12.14
- PyTorch：2.6.0+cu124
- CUDA：12.4
- vLLM：0.8.5.post1
- Transformers：4.51.1
- 虚拟环境：`/home/asus/venvs/vllm`

本机适合：

- Qwen2.5-0.5B smoke test
- HF/vLLM 基线
- 低并发实验
- 2048/4096 上下文的小规模实验

Kaggle 环境（2026-09-27 实测，1.5B 全矩阵）：

- GPU：Tesla T4 16GB ×2（Turing / **SM 7.5**）
- 容器 hostname：`6b937f1eb901`（本次 22 个 run 全部相同 → 同一次 session）
- Python 3.12.13 / PyTorch 2.6.0+cu124 / CUDA 12.4 / vLLM 0.8.5.post1
- **Transformers 4.51.1**（从 5.0.0 降级；**metadata 里没有这个字段**，见下）
- **`dtype=half`**（T4 无 BF16）— 20/20 个 vLLM run 一致
- **引擎 V0**（T4 不支持 V1）— **只来自操作者陈述，产物无法自证**，见下
- **attention backend：XFormers**（T4 上 FA2 不适用）
- `gpu_memory_utilization=0.85`（全部 run 一致）
- **GPU 遥测：无**（Kaggle 侧一个都没采集）
- 用于 1.5B、7B、量化和双卡 Tensor Parallel 实验
- Kaggle 执行方式见 `docs/runbook_three_models.md` 第 5 节

> **环境记录的两个缺陷**（详见报告第 13.4 节）：
> 1. `results/raw/env-kaggle-1.5b.json` 记的是 `transformers: 5.0.0`，
>    与实际使用的降级后版本 **4.51.1 矛盾**，**该快照不可信**；
> 2. 每次 run 的 metadata **不记 `transformers` 也不记 `VLLM_USE_V1`**，
>    所以"哪个 run 用哪个版本""哪些 run 跑在 V0"都无法从产物确认。
>
> 第 2 条已于 2026-09-27 修补（`scripts/common.py` 的 `env_vars` 字段 + `transformers` 版本），
> **但只对未来的 run 生效**。

7B FP16 不作为本机验收条件。7B 实验优先使用已经验证可加载的 AWQ/GPTQ checkpoint。

## 3. Directory Structure

```text
vllm-benchmark-lab/
├── README.md
├── project_summary.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   ├── qwen_0.5b.yaml
│   ├── qwen_1.5b.yaml
│   └── qwen_7b.yaml
├── scripts/
│   ├── common.py                # 共享 workload / 计时 / JSONL helper / 环境元数据
│   ├── run_hf.py                # HF generate 逐请求
│   ├── run_vllm.py              # vLLM 离线 LLM.generate
│   ├── benchmark_http.py        # OpenAI 兼容 server 压测（per-request TTFT/TPOT）
│   ├── make_prompts.py          # 生成精确 token 长度的 prompt
│   ├── summarize_results.py     # 聚合 raw JSONL
│   ├── collect_gpu_metrics.py   # nvidia-smi 时间序列
│   └── benchmark.py             # YAML 驱动的统一控制器
├── experiments/                 # 每个实验只留 README（说明 + 命令 + 结果）
│   ├── hf_vs_vllm/
│   ├── concurrency/
│   ├── context_length/
│   ├── prefix_caching/
│   ├── chunked_prefill/
│   └── quantization/
├── results/
│   ├── raw/          (.gitkeep) # 原始 JSONL —— 本地生成，不提交
│   ├── processed/    (.gitkeep) # 聚合表   —— 本地生成，不提交
│   └── plots/        (.gitkeep) # 图表     —— 本地生成，不提交
└── docs/
    ├── benchmark_report.md      # 实验报告：Part I 本机 0.5B + Part II Kaggle 1.5B
    ├── runbook_three_models.md  # 三模型操作手册 + 坑 1–13
    └── kaggle_runbook.md        # 早期 Kaggle 笔记（已归档为历史文档）
```

## 4. Implemented Scripts

### `scripts/run_hf.py`

使用 Transformers `AutoModelForCausalLM` 和 `generate` 逐请求运行。

记录：

- input tokens
- output tokens
- latency
- TPOT 近似值
- output tok/s

本地单卡使用显式 `.to("cuda")`，不依赖 `accelerate`。

### `scripts/run_vllm.py`

使用 vLLM 离线 `LLM.generate` 批量运行。

支持参数：

- `--max-model-len`
- `--max-num-seqs`
- `--max-num-batched-tokens`（chunked prefill A/B 必需）
- `--enable-prefix-caching`
- `--enable-chunked-prefill`
- `--kv-cache-dtype`
- `--quantization`
- `--tensor-parallel-size`
- `--gpu-memory-utilization`
- `--requests`
- `--max-tokens`
- `--prompts`
- `--include-text` / `--include-prompts`

**注意**：`enable_prefix_caching` 是显式传值的（包括 False），因为 V1 下默认是 True，
省略参数会静默变成"开"。见 `docs/runbook_three_models.md` 坑 2。

### `scripts/make_prompts.py`

**新增**。生成**精确 token 长度**的 prompt，四种模式：`distinct` / `shared-prefix` /
`random-prefix` / `mixed`。带 `--check` 复核、`--strict` 保证长度精确，
并输出 `*.manifest.json` 记录 workload 分布（填报告第 3 节用）。

**生成的 `prompts_*.txt` 与 `.manifest.json` 都不进版本库**（`.gitignore` 已排除）：
它们是实验输入数据，用固定 seed=42 和 `docs/runbook_three_models.md` 第 3.2 节的命令
可逐字节重建，所以仓库里只留生成命令，不留产物。

### `scripts/benchmark_http.py`

**新增**。走 OpenAI 兼容 server 压测，输出 **per-request** 的 `ttft_ms` / `tpot_ms` /
`latency_ms` 与 mean/p50/p95/p99。这是唯一能拿到真实 TTFT/TPOT/百分位的路径
（离线 `LLM.generate` 只有 batch 级数字）。

### `scripts/summarize_results.py`

**新增**。把 `results/raw/*.jsonl` 聚合成报告用的 markdown / CSV，
并自动给每个 run 打上 `latency_mode=batch|per-request|single` 标签，
对 batch run 隐藏无意义的百分位。

### `scripts/benchmark.py`

读取 YAML 配置，执行 warmup、重复实验，并将结果聚合到 `results/processed/`。

### `scripts/collect_gpu_metrics.py`

通过 `nvidia-smi` 周期性采集：

- GPU utilization
- memory used
- memory total
- power draw
- temperature
- SM clock

GPU 采集器需要在 benchmark 运行期间同时启动，但不需要永久运行。建议覆盖模型加载、warmup 和正式推理阶段。

## 5. Important Measurement Semantics

### `requests`

本轮总共提交的请求数量。

例如：

```text
requests=64
```

表示总共处理 64 个请求。

### `max_num_seqs`

vLLM 同时最多调度的序列数。

例如：

```text
requests=64
max_num_seqs=16
```

表示总共 64 个请求，最多以 16 条序列为一个调度批次，可能分成多个批次执行。

### `max_model_len`

单请求允许的最大上下文长度上限，不代表实际 prompt 已经达到这个长度。

如果 prompt 实际只有 10 tokens，即使配置 `max_model_len=4096`，也不是真正的 4096-token 上下文实验。

## 6. Important Finding: Offline vLLM JSONL Values

当前 `run_vllm.py` 的核心逻辑是：

```python
outputs = llm.generate(prompts, sampling, use_tqdm=False)
elapsed_ms = (time.perf_counter() - start) * 1000
```

因此，当前离线 vLLM runner 测量的是整个 batch 的耗时。

同一个 batch 内的每条记录都会得到相同的：

- `latency_ms`
- `output_tok_per_s`

这就是 `concurrency-16.jsonl` 中所有请求数值相同的原因，属于当前实现的预期行为。

这些字段应解释为：

> 整个批次的 batch latency 和 batch output throughput

不能解释为每个请求的真实独立延迟。

HF runner 是逐请求运行，所以 HF JSONL 中不同请求的 latency 通常不同。

当前离线 vLLM 模式适合观察：

- 总批次耗时
- 总输出吞吐
- batch 调度效果
- 显存和 GPU 利用率

当前离线 vLLM 模式不适合精确测量：

- 每请求 TTFT
- 每请求 TPOT
- HTTP 排队延迟
- p95/p99 请求延迟

精确 TTFT、TPOT、p95 和 p99 应使用 OpenAI-compatible server + HTTP client 压测。

## 7. Existing Results

### 7.1 本机 0.5B（2026-09-26，21 个 run）

**本节开头原本记录的是 2026-09-25 的早期 smoke run，那些文件已被 2026-09-26 的全矩阵取代**
（`hf-smoke.jsonl`、`vllm-smoke.jsonl`、`concurrency-16.jsonl` 均已不存在，
新命名规范见 `docs/runbook_three_models.md` 第 3.3 节）。

**当前以 `results/raw/local-0.5b-*.jsonl` 为准（21 个 run），
完整分析见 `docs/benchmark_report.md` Part I（第 1–12 节）。**

> 下表中的文件名是**本地数据文件**，已被 `.gitignore` 排除、不进版本库；
> 聚合表 `results/processed/summary-local-0.5b.{md,csv}` 同样不提交，本地用
> `scripts/summarize_results.py` 重建（命令见 README）。

| 实验 | 文件 |
| --- | --- |
| E1 HF vs vLLM | `local-0.5b-hf-single`、`local-0.5b-vllm-single`、`local-0.5b-vllm-batch16` |
| E2 并发 | `local-0.5b-concurrency-{16,32,64}` |
| E3 上下文 | `local-0.5b-context-{512,2048,4096,8192,16384}` |
| E4 Prefix caching | `local-0.5b-prefix-{shared,random}-{off,on}` |
| E5 Chunked prefill | `local-0.5b-chunked-{off,on,v1ref}` |
| E6 量化 | `local-0.5b-awq`、`local-0.5b-kvfp8` |
| Server 模式 | `local-0.5b-http-concurrency-16`、`gpu-local-0.5b-http-16` |

两条最重要的结论：

```text
E1a 单请求：HF 3675.41 ms / 17.41 tok/s  vs  vLLM 986.91 ms / 64.85 tok/s   → vLLM 快 3.7 倍
E1b 批处理：vLLM 1 条 64.85 tok/s → 16 条 1075.51 tok/s                    → 吞吐 ×16.6，延迟几乎不变
```

### 7.2 Kaggle 1.5B（2026-09-27，22 个文件）

**以 `results/raw/kaggle-1.5b-*.jsonl` 为准，完整分析见
`docs/benchmark_report.md` Part II（第 13–23 节）。**

> **重要前提**：T4 不支持 vLLM V1，所以**除 HF 外 20 个 run 全部跑在 V0 引擎上**。
> 好处是 Kaggle 内部的对照都是同引擎对照（量化一节因此首次可归因）；
> 代价是与本机（V1）比较多了一层引擎差异。详见报告第 13.3 节。

| 实验 | 文件 |
| --- | --- |
| E1 HF vs vLLM | `kaggle-1.5b-hf-single`、`kaggle-1.5b-vllm-single`、`kaggle-1.5b-vllm-batch16` |
| E2 并发 | `kaggle-1.5b-concurrency-{16,64,128,256}` |
| E3 上下文 | `kaggle-1.5b-context-{512,2048,4096,8192,16384}` |
| E4 Prefix caching | `kaggle-1.5b-prefix-{shared,random}-{off,on}` |
| E5 Chunked prefill | `kaggle-1.5b-chunked-{off,on}`（**没有 v1ref 臂**，T4 无 V1）|
| E6 量化 | `kaggle-1.5b-awq`（GPTQ 未执行）|
| E7 并行 | `kaggle-1.5b-tp1`、`kaggle-1.5b-tp2` |
| E8 Server 模式 | `kaggle-1.5b-http-concurrency-16` —— **失败实验**（64/64 连接错误）|

**三条主要结论：**

```text
E2  并发天花板：max_num_seqs ≥ 并发请求数后不再有收益
                16→64 ×1.42，之后 64→128 −1.4%、128→256 −2.0%
E6  量化：AWQ +13.3% 吞吐（延迟 ×0.883）—— 本项目第一个可归因的量化结论
E7  并行：TP=2 快 24.8%（推翻 runbook 预设），但每卡吞吐掉 37.6%
```

**未执行**：Kaggle 7B、E7b 双实例、GPTQ、fp8 KV（**T4 硬件不支持**）、
以及**全部 per-request 指标**（TTFT/TPOT/p95/p99，因 E8 失败）。

**已知异常（未解释）**：E4 的 `prefix-shared-on` 比 `prefix-shared-off` **慢 2.49 倍**，
与本机同实验的 +21.9% **方向相反**。相邻 run 的 pid 已排除机器漂移，
但原因未定，**该节结论记为"空"**，见报告第 18 节。


这些仍然是整个离线 batch 的测量值（`latency_mode=batch`），不是 per-request 延迟。

## 8. How To Observe GPU Metrics

### Terminal 1: GPU collector

```bash
cd ~/vllm-benchmark-lab

/home/asus/venvs/vllm/bin/python scripts/collect_gpu_metrics.py \
  --run-id concurrency-16 \
  --output results/raw/gpu-concurrency-16.jsonl \
  --interval 0.5 \
  --duration 180
```

### Terminal 2: Benchmark

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/asus/venvs/vllm/bin/python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 64 \
  --max-num-seqs 16 \
  --max-model-len 2048 \
  --max-tokens 64 \
  --gpu-memory-utilization 0.75 \
  --output results/raw/concurrency-16.jsonl
```

GPU collector does not need to stay open permanently. It only needs to cover:

1. model loading
2. compilation/warmup
3. formal inference
4. a few seconds after inference

Recommended duration on the local machine: 180 seconds.

### Optional live monitoring

```bash
watch -n 0.5 nvidia-smi
```

## 9. How To Interpret GPU Metrics

### GPU utilization

- 0%-30%: GPU may be waiting for CPU, input preparation or insufficient workload
- 50%-80%: normal workload
- 80%-100%: high GPU utilization

GPU utilization should be interpreted together with throughput and latency. Higher utilization is not automatically better.

### Memory usage

Calculate:

```text
memory.used / memory.total * 100%
```

Suggested interpretation:

- below 70%: comfortable headroom
- 70%-90%: getting tight
- above 90%: close to the limit
- near 100% or CUDA OOM: configuration is too large

### OOM detection

Check process exit code:

```bash
echo $?
```

Search logs for:

```text
CUDA out of memory
OutOfMemoryError
Engine core initialization failed
```

An OOM experiment should be kept as a failed experiment with a clear error message, not deleted.

## 10. Concurrency Experiment

Recommended first comparison:

```text
requests=64, max_num_seqs=16
requests=64, max_num_seqs=64
```

This keeps total work fixed and changes the maximum scheduling batch size.

### `max_num_seqs=16`

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/asus/venvs/vllm/bin/python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 64 \
  --max-num-seqs 16 \
  --max-model-len 2048 \
  --max-tokens 64 \
  --gpu-memory-utilization 0.75 \
  --output results/raw/concurrency-16.jsonl
```

### `max_num_seqs=64`

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/asus/venvs/vllm/bin/python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 64 \
  --max-num-seqs 64 \
  --max-model-len 2048 \
  --max-tokens 64 \
  --gpu-memory-utilization 0.75 \
  --output results/raw/concurrency-64.jsonl
```

On an 8GB RTX 4060, start with `max_num_seqs=16` and `32` if `64` causes OOM.

## 11. Prefix Caching Experiment

Do not edit `scripts/common.py` for one experiment. Pass a dedicated prompt file with `--prompts`.

Example file:

```text
You are a helpful assistant. Answer clearly and briefly. Explain batching in LLM inference.
You are a helpful assistant. Answer clearly and briefly. Explain KV cache in LLM inference.
You are a helpful assistant. Answer clearly and briefly. Explain TTFT in LLM inference.
You are a helpful assistant. Answer clearly and briefly. Explain vLLM scheduling in LLM inference.
```

Save it as:

```text
experiments/prefix_caching/prompts_repeated.txt
```

Run prefix caching off and on with the same prompt file:

```bash
# off
python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts experiments/prefix_caching/prompts_repeated.txt \
  --requests 16 \
  --max-num-seqs 16 \
  --output results/raw/prefix-off.jsonl
```

```bash
# on
python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts experiments/prefix_caching/prompts_repeated.txt \
  --requests 16 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --output results/raw/prefix-on.jsonl
```

Only explain caching gains when requests really share a prefix.

## 12. True Context-Length Experiment

The default prompts are only about 10-12 tokens, so they do not test long context.

Prepare these files:

```text
experiments/context_length/prompts_512.txt
experiments/context_length/prompts_2048.txt
experiments/context_length/prompts_4096.txt
```

A prompt generator can be created as:

```python
from pathlib import Path
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
base = (
    "This is a controlled language model benchmarking document. "
    "The purpose is to evaluate inference latency, throughput, memory usage, "
    "attention behavior, and scheduling efficiency. "
)

for target_tokens in [512, 2048, 4096]:
    text = base * (target_tokens // 30 + 10)
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
    Path(f"prompts_{target_tokens}.txt").write_text(prompt + "\n", encoding="utf-8")
    print(target_tokens, len(tokenizer.encode(prompt, add_special_tokens=False)))
```

Run it from `experiments/context_length/` and verify the actual token count.

### 2048-token experiment

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/asus/venvs/vllm/bin/python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts experiments/context_length/prompts_2048.txt \
  --requests 16 \
  --max-num-seqs 16 \
  --max-model-len 2048 \
  --max-tokens 64 \
  --output results/raw/context-2048.jsonl
```

### 4096-token experiment

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
/home/asus/venvs/vllm/bin/python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts experiments/context_length/prompts_4096.txt \
  --requests 4 \
  --max-num-seqs 4 \
  --max-model-len 4096 \
  --max-tokens 64 \
  --output results/raw/context-4096.jsonl
```

On an 8GB GPU, start with fewer requests for 4096-token prompts.

Verify actual token counts:

```bash
grep -o '"input_tokens":[0-9]*' results/raw/context-2048.jsonl
grep -o '"input_tokens":[0-9]*' results/raw/context-4096.jsonl
```

## 13. Recommended Next Steps

上面那一版（2026-09-25 写的 6 条）已全部完成。2026-09-27 跑完 Kaggle 1.5B 后，
待办按优先级重排如下，详细版在 `docs/runbook_three_models.md` 第 10 节。

**已完成（本轮）：**

- ~~把 `VLLM_USE_V1` 写进 JSONL~~ → **已修补** `environment_metadata()`，新增 `env_vars`
  与 `transformers` 字段。**只对未来的 run 生效**，已有产物仍缺。
- ~~给 E6 补一条对齐的 FP16 基线~~ → **Kaggle 1.5B 上做到了**：同引擎（V0）、
  同 `gpu_memory_utilization=0.85`、同 prompt，唯一变量是权重位宽 → **AWQ +13.3% 可归因**。
- ~~E2 向上扫 `max_num_seqs`（128/256）~~ → **Kaggle 上扫完了**：
  **天花板 = 并发请求数**，64 之后没有收益（但 8GB 的 **OOM 边界仍未找到**）。
- ~~开始 Kaggle 部分~~ → **1.5B 全矩阵完成**，三模型矩阵 2/3。

**待办（按优先级）：**

1. **查清 E4 的 prefix caching 异常**（报告第 18 节）。T4 上 shared-on 比 shared-off
   **慢 2.49 倍**，与本机 +21.9% 直接矛盾。这是**唯一一条两台机器结论相反的实验**，
   不查清则 E4 在 Kaggle 侧等于没有结论。方案：同 workload 跑 3 次 + 保存 vLLM
   启动日志（确认 `enable_prefix_caching` 实际取值与所选 attention backend）。
2. **在 Kaggle 上拿到 per-request 指标**（报告第 22.3 节的 cell 内 subprocess 方案）。
   目前 Kaggle 侧**零 TTFT/TPOT/p95**，E4/E5 的尾延迟问题在两台机器上都没答；
   也做不了 Part I 第 5.3 节那种 server/离线互验。
3. **给 Kaggle 补 GPU 遥测**。E2/E6 的显存问题全部无解（"量化省多少显存"仍是空白）。
4. **7B（AWQ vs GPTQ）+ E7b 两个独立单卡实例**。E7b 能验证报告 21.2 的推算
   ——TP=2 每卡效率掉 37.6%，两个独立实例是否真的更优。
5. **给关键配置加重复实验**（每个 ≥3 次）。本轮已测出**噪声底 ±1.2%**
   （报告 13.5，来自 `tp1` 与 `concurrency-64` 这对意外重复），
   但大多数臂仍只有单次；±1.2% 以内的差异（如 E2 的 64→128）目前无法解释。
6. **修 `docs/kaggle_runbook.md`**：该文档与 `runbook_three_models.md` 第 5 节重复且更旧，
   现已标注为历史文档，建议合并。
7. **本机 0.5B 的 OOM 边界仍未找到**（Part I 遗留）。Kaggle 证明的是"没有收益"，
   不是"会 OOM"，两件事不要混。

