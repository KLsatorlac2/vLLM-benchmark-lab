# vLLM Benchmark Lab

一个面向单卡和双卡环境的 LLM 推理性能实验室，用于把 HF `generate` 与 vLLM 的推理差异转换成可复现的数据。

## 目标

- 比较 Transformers/HuggingFace 与 vLLM 的延迟和吞吐。
- 研究并发数、上下文长度、prefix caching、chunked prefill 对性能的影响。
- 比较 FP16/BF16、AWQ、GPTQ 和 KV cache dtype 的显存与速度权衡。
- 形成可回溯的 JSONL 原始结果、聚合数据、图表和实验报告。

## 资源策略

| 环境 | 用途 |
| --- | --- |
| RTX 4060 8GB | Qwen2.5-0.5B smoke test、HF/vLLM 小规模基线、低并发实验 |
| Kaggle T4 16GB x2 | 1.5B/7B、并发扫描、量化和双卡实验 |

7B FP16 不作为本机验收条件。7B 配置默认量化优先，具体 AWQ/GPTQ checkpoint 必须先验证能否加载。

## 安装

建议复用已有 CUDA 环境，不让 `pip` 自动替换 PyTorch/vLLM：

```bash
cd ~/vllm-benchmark-lab
/home/asus/venvs/vllm/bin/python -m pip install -r requirements.txt
```

检查环境：

```bash
/home/asus/venvs/vllm/bin/python -c "import torch, vllm, transformers; print(torch.__version__, torch.version.cuda, vllm.__version__, transformers.__version__); print(torch.cuda.device_count())"
```

## 最小运行链

模型在 Hugging Face cache 中可用时，本机先执行：

```bash
cd ~/vllm-benchmark-lab
/home/asus/venvs/vllm/bin/python scripts/run_hf.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 2 --max-tokens 32 \
  --output results/raw/hf-smoke.jsonl

/home/asus/venvs/vllm/bin/python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 2 --max-tokens 32 \
  --max-model-len 2048 \
  --output results/raw/vllm-smoke.jsonl
```

使用统一控制器：

```bash
/home/asus/venvs/vllm/bin/python scripts/benchmark.py \
  --config configs/qwen_0.5b.yaml
```

先检查命令但不加载模型：

```bash
/home/asus/venvs/vllm/bin/python scripts/benchmark.py \
  --config configs/qwen_0.5b.yaml --dry-run
```

## 指标口径

- `TTFT`：请求发送到第一个输出 token 的时间。
- `TPOT`：首 token 之后，平均生成一个输出 token 的时间。
- `output_tok_per_s`：真实生成 token 数除以耗时。
- `req/s`：完成请求数除以测试窗口时间。
- 延迟至少报告 mean、p50、p95、p99。
- GPU 指标来自 `nvidia-smi` 时间序列。
- KV cache usage 只有 vLLM 明确提供时才记录，不能用总显存占用替代。

当前离线 `LLM.generate` runner 适合模型级吞吐基线；精确 TTFT/TPOT 和 HTTP 并发测量应使用 OpenAI-compatible server 路径补充。

## 实验顺序

> 三模型（0.5B 本机 / 1.5B Kaggle / 7B Kaggle）的完整操作流程、命令和已知陷阱见
> `docs/runbook_three_models.md`。开工前先读该文档第 1 节的五个坑。

1. `experiments/hf_vs_vllm/`：0.5B 单请求和小批量基线。
2. `experiments/concurrency/`：`max_num_seqs=16/64/256`。
3. `experiments/context_length/`：`max_model_len=4096/8192/16384`。
4. `experiments/prefix_caching/`：重复前缀与随机前缀。
5. `experiments/chunked_prefill/`：长输入、短输出和混合 workload。
6. `experiments/quantization/`：FP16/BF16、AWQ、GPTQ、KV cache dtype。

每个实验只保存配置、命令和说明，不复制 runner 逻辑。原始 JSONL 放在 `results/raw/`，聚合结果放在 `results/processed/`，图表放在 `results/plots/`。

## 实验结果（Qwen2.5-0.5B 本机，2026-09-26）

**状态：本机 0.5B 全矩阵（E1–E6）已完成，21 个 run 全部 `status=ok`。
Kaggle 1.5B / 7B / TP=2 尚未执行，E7（并行）因本机只有 1 张卡跳过。**

完整分析见 **[`docs/benchmark_report.md`](docs/benchmark_report.md)**。

> **实验数据不进版本库。** `results/raw/`（原始 JSONL）和 `results/processed/`（聚合表）
> 在 `.gitignore` 里被排除，仓库只保留目录结构（`.gitkeep`）。
> 聚合表在本地一条命令重建：
>
> ```bash
> python scripts/summarize_results.py results/raw/local-0.5b-*.jsonl \
>   --gpu results/raw/gpu-local-0.5b-http-16.jsonl \
>   --output-md results/processed/summary-local-0.5b.md \
>   --output-csv results/processed/summary-local-0.5b.csv
> ```
>
> 因此本 README 与报告里的每个数字都标注了它来自哪个 `results/raw/` 文件，
> 便于在有数据的机器上回溯核对。

环境：RTX 4060 Laptop 8GB（Ada）、Python 3.12.14、torch 2.6.0+cu124、vLLM 0.8.5.post1、
`dtype=auto`→bfloat16、单卡。所有实验 `temperature=0`、`seed=42`、`max_tokens=64`。

### 主要结论

| # | 结论 | 证据（`results/raw/`）|
| --- | --- | --- |
| 1 | 单请求下 **vLLM 比 HF 快 3.7 倍**（986.91 ms vs 3675.41 ms，吞吐 64.85 vs 17.41 tok/s）| `hf-single` vs `vllm-single` |
| 2 | vLLM 自己从 1 条加到 16 条：**吞吐 ×16.6，batch 延迟几乎不变**（×0.965）| `vllm-single` → `vllm-batch16` |
| 3 | 并发 16→64：**吞吐 ×1.72、端到端延迟 ×0.58**，但边际收益递减（32→64 只有 ×1.275）| `concurrency-16/32/64` |
| 4 | 总 prefill 固定 8192 token 时，**延迟与单条长度基本无关**（1374–1545 ms，×1.125）| `context-512/2048/4096/8192` |
| 5 | 长上下文的吞吐暴跌是 **batch 宽度效应，不是长度效应**（batch 16→1）| 同上 |
| 6 | **prefix caching 只在真有共享前缀时有用**：shared **+21.9%**、random **−1.2%** | `prefix-{shared,random}-{off,on}` |
| 7 | **chunked prefill 在离线 batch 下是负收益（−11.7%）**；V1 的切块比 V0 快 11.2% | `chunked-{off,on,v1ref}` |
| 8 | 量化（AWQ / fp8 KV）相对 FP16 基线都是 −13%，**但归因不了**（引擎版本与显存配额未对齐）| `awq`、`kvfp8` vs `vllm-batch16` |
| 9 | server 与离线两条独立路径在 6% 内吻合（941.23 vs 1005.55 tok/s），**互相验证** | `http-concurrency-16` vs `concurrency-16` |

### 本轮暴露的三个记录缺陷

1. **`VLLM_USE_V1` 没有写进 JSONL**，导致 AWQ 臂事后无法确认跑在 V0 还是 V1 上——
   而 V0/V1 之间本身就有约 10% 的差距。**这是最该先修的**，见 `environment_metadata()`。
2. **除 server run 外没有 GPU 遥测**，E2/E3/E6 的显存全部无数据，"量化省了多少显存"无从回答。
3. **chunked prefill 只测了离线 batch**，而它的卖点是短请求尾延迟——设计上就观测不到。

两个实测报错（V0 的 `max_num_batched_tokens ≥ max_model_len`、V1 不支持 fp8 KV cache）
已记录在 [`docs/runbook_three_models.md`](docs/runbook_three_models.md) 坑 6 / 坑 7。

## 目录

```text
vllm-benchmark-lab/
├── README.md
├── requirements.txt
├── configs/
├── scripts/
├── experiments/
├── results/
└── docs/
```

## 常见限制

- 量化不是单纯的 dtype 开关，需要匹配实际量化 checkpoint。
- T4 不等于本机 4060；跨硬件结论必须分别标注。
- Tensor Parallel、两个单卡实例和单卡运行不是同一种实验，结果不能直接混排。
- OOM、未支持的 dtype 或缺少 checkpoint 都要作为失败实验保留，并写明原因。

## 简历描述模板

> 构建可复现的 vLLM 推理基准实验室，统一比较 Transformers 与 vLLM 在不同模型、并发、上下文长度、prefix caching、chunked prefill 和量化策略下的 TTFT、TPOT、吞吐、显存和 GPU 利用率；支持本地单卡与 Kaggle T4 双卡实验，并通过 JSONL、聚合脚本和可视化报告完成结果追溯。
