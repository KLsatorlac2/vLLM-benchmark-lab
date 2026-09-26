# vLLM Benchmark Lab Project Summary

## 1. Project Overview

项目名称：`vllm-benchmark-lab`

项目目标：把 LLM 推理优化从主观感受转化为可复现实验数据，比较 HuggingFace Transformers 与 vLLM 在不同模型、并发、上下文长度、缓存策略和量化策略下的性能差异。

实验矩阵（E1–E7）与三模型分工的完整定义见 `docs/runbook_three_models.md` 第 2 节。

当前进度（2026-09-26）：

1. HF vs vLLM：Qwen2.5-0.5B — **已完成**（单请求 + 批处理两步拆开）
2. 并发：`max_num_seqs` = 16、32、64 — **已完成**（128/256 未扫，OOM 边界未知）
3. 上下文长度：512、2048、4096、8192、16384 — **已完成**
4. Prefix caching：开启/关闭 × 共享前缀/随机前缀 — **已完成**（四臂对照）
5. Chunked prefill：开启/关闭（V0）+ V1 参考臂 — **已完成**，但只有离线 batch 数字
6. 量化：AWQ、fp8 KV cache — **已完成**；GPTQ **未执行**
7. 并行：TP=1 vs TP=2 — **未执行**（本机单卡，属 Kaggle 部分）
8. Kaggle 1.5B / 7B — **未执行**

结论见 `docs/benchmark_report.md`。
聚合表（`results/processed/summary-local-0.5b.md`）**不进版本库**，本地用
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

Kaggle 环境：

- T4 16GB x2
- 用于 1.5B、7B、量化、双卡 Tensor Parallel 和更大并发实验
- Kaggle 执行方式见 `docs/kaggle_runbook.md`

7B FP16 不作为本机验收条件。7B 实验优先使用已经验证可加载的 AWQ/GPTQ checkpoint。

## 3. Directory Structure

```text
vllm-benchmark-lab/
├── README.md
├── project_summary.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── qwen_0.5b.yaml
│   ├── qwen_1.5b.yaml
│   └── qwen_7b.yaml
├── scripts/
│   ├── common.py
│   ├── run_hf.py
│   ├── run_vllm.py
│   ├── benchmark.py
│   └── collect_gpu_metrics.py
├── experiments/
│   ├── hf_vs_vllm/
│   ├── concurrency/
│   ├── context_length/
│   ├── prefix_caching/
│   ├── chunked_prefill/
│   └── quantization/
├── results/
│   ├── raw/
│   ├── processed/
│   └── plots/
└── docs/
    ├── kaggle_runbook.md
    └── benchmark_report.md
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

## 7. Existing Local Results

**本节记录的是 2026-09-25 的早期 smoke run，这些文件已被 2026-09-26 的全矩阵实验取代**
（`hf-smoke.jsonl`、`vllm-smoke.jsonl`、`concurrency-16.jsonl` 均已不存在，
新命名规范见 `docs/runbook_three_models.md` 第 3.3 节，带 `local-0.5b-` 前缀以免与 Kaggle 结果相撞）。

**当前实际的实验结果以 `results/raw/local-0.5b-*.jsonl` 为准（21 个 run），
完整分析见 `docs/benchmark_report.md`。**

> 下表中的文件名是**本地数据文件**，已被 `.gitignore` 排除、不进版本库；
> 聚合表 `results/processed/summary-local-0.5b.{md,csv}` 同样不提交，本地用
> `scripts/summarize_results.py` 重建（命令见 README）。

已完成（2026-09-26，Qwen2.5-0.5B，本机 RTX 4060）：

| 实验 | 文件 |
| --- | --- |
| E1 HF vs vLLM | `local-0.5b-hf-single`、`local-0.5b-vllm-single`、`local-0.5b-vllm-batch16` |
| E2 并发 | `local-0.5b-concurrency-{16,32,64}` |
| E3 上下文 | `local-0.5b-context-{512,2048,4096,8192,16384}` |
| E4 Prefix caching | `local-0.5b-prefix-{shared,random}-{off,on}` |
| E5 Chunked prefill | `local-0.5b-chunked-{off,on,v1ref}` |
| E6 量化 | `local-0.5b-awq`、`local-0.5b-kvfp8` |
| Server 模式 | `local-0.5b-http-concurrency-16`、`gpu-local-0.5b-http-16` |

**尚未执行**：Kaggle 1.5B / 7B、TP=2、双实例、GPTQ、server 模式下的 E5 尾延迟。

两条最重要的结论（详见 `docs/benchmark_report.md`）：

```text
E1a 单请求：HF 3675.41 ms / 17.41 tok/s  vs  vLLM 986.91 ms / 64.85 tok/s   → vLLM 快 3.7 倍
E1b 批处理：vLLM 1 条 64.85 tok/s → 16 条 1075.51 tok/s                    → 吞吐 ×16.6，延迟几乎不变
```

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

上面这一版（2026-09-25 写的 6 条）已全部完成，见第 7 节与 `docs/benchmark_report.md`。
当前待办按优先级重排如下，详细版在 `docs/runbook_three_models.md` 第 10 节：

1. **把 `VLLM_USE_V1` 写进 JSONL**（`scripts/common.py` 的 `environment_metadata()`）。
   现在事后无法判断任一 run 跑在 V0 还是 V1 上，而 V0/V1 之间本身就有约 10% 的差距
   （E5 实测），已经导致 AWQ 臂无法归因。
2. **给 E2/E3/E6 补 GPU 遥测**。目前只有 server run 有采集，量化"省显存"这条完全没有数据。
3. **给 E6 补一条对齐的 FP16 基线**：同引擎版本、同 `gpu_memory_utilization`、同 prompt。
   现在的 −13% 归因不了。
4. **用 server 模式重测 E5**（chunked prefill 的短请求 TTFT）。
   离线 batch 设计上就观察不到它的卖点，现在这一节实际上没有有效结论。
5. **E2 继续向上扫 `max_num_seqs`（128/256）**，找到 8GB 的 OOM 边界。
6. **开始 Kaggle 部分**：1.5B 全矩阵 + TP=1 vs TP=2，然后 7B 的 AWQ vs GPTQ。
   三模型矩阵目前只完成 1/3。
7. **给关键配置加重复实验**（每个 ≥3 次），否则 ±10% 以内的差异无法与噪声区分。
