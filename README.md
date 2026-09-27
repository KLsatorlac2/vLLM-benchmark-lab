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
> `docs/runbook_three_models.md`。开工前按环境对号入座读坑：
> **本机读第 1 节的坑 1–7，Kaggle 读第 1.1 节的坑 8–13**（T4 只能用 V0、必须 `--dtype half`、
> server 不能后台常驻等六个问题都会静默改变配置口径）。

1. `experiments/hf_vs_vllm/`：0.5B 单请求和小批量基线。
2. `experiments/concurrency/`：`max_num_seqs=16/64/256`。
3. `experiments/context_length/`：`max_model_len=4096/8192/16384`。
4. `experiments/prefix_caching/`：重复前缀与随机前缀。
5. `experiments/chunked_prefill/`：长输入、短输出和混合 workload。
6. `experiments/quantization/`：FP16/BF16、AWQ、GPTQ、KV cache dtype。

每个实验只保存配置、命令和说明，不复制 runner 逻辑。原始 JSONL 放在 `results/raw/`，聚合结果放在 `results/processed/`，图表放在 `results/plots/`。

## 实验结果

**状态：三模型矩阵完成 2/3。**

| 环境 | 模型 | 状态 |
| --- | --- | --- |
| 本机 RTX 4060 8GB | Qwen2.5-0.5B | ✅ 全矩阵 E1–E6（21 个 run，全部 `status=ok`）；E7 因单卡跳过 |
| Kaggle T4 ×2 | Qwen2.5-1.5B | ✅ 全矩阵 E1–E7（22 个文件：20 个 vLLM run + 1 个 HF run + 1 个失败的 server run）；**E8 HTTP server 失败** |
| Kaggle T4 ×2 | Qwen2.5-7B | ❌ 未执行 |

完整分析见 **[`docs/benchmark_report.md`](docs/benchmark_report.md)**——
**Part I（第 1–12 节）** 是本机 0.5B，**Part II（第 13–23 节）** 是 Kaggle 1.5B。
两部分的绝对数字**不可横向比较**（GPU / 模型 / dtype / 引擎四个变量同时不同），
原因见报告第 23.1 节。

> **实验数据不进版本库。** `results/raw/`（原始 JSONL）和 `results/processed/`（聚合表）
> 在 `.gitignore` 里被排除，仓库只保留目录结构（`.gitkeep`）。
> 聚合表在本地一条命令重建：
>
> ```bash
> # 本机 0.5B
> python scripts/summarize_results.py results/raw/local-0.5b-*.jsonl \
>   --gpu results/raw/gpu-local-0.5b-http-16.jsonl \
>   --output-md results/processed/summary-local-0.5b.md \
>   --output-csv results/processed/summary-local-0.5b.csv
>
> # Kaggle 1.5B
> python scripts/summarize_results.py results/raw/kaggle-1.5b-*.jsonl \
>   --output-md results/processed/summary-kaggle-1.5b.md \
>   --output-csv results/processed/summary-kaggle-1.5b.csv
> ```
>
> 因此本 README 与报告里的每个数字都标注了它来自哪个 `results/raw/` 文件，
> 便于在有数据的机器上回溯核对。

### Part I：本机 RTX 4060 / Qwen2.5-0.5B（2026-09-26）

环境：RTX 4060 Laptop 8GB（Ada）、Python 3.12.14、torch 2.6.0+cu124、vLLM 0.8.5.post1、
`dtype=auto`→bfloat16、单卡、引擎 V1。所有实验 `temperature=0`、`seed=42`、`max_tokens=64`。

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

### Part II：Kaggle T4 ×2 / Qwen2.5-1.5B（2026-09-27）

环境：Tesla T4 16GB ×2（Turing / **SM 7.5**）、Python 3.12.13、torch 2.6.0+cu124、
vLLM 0.8.5.post1、**`dtype=half`**、**引擎 V0**、**XFormers**、`gpu_memory_utilization=0.85`。

> **T4 不支持 vLLM V1**，所以除 HF 外 20 个 run **全部退回 V0**（原因与影响见报告第 13.3 节）。
> 这反而让 Kaggle 内部的对照变成同引擎对照——**量化一节因此首次可归因**。

| # | 结论 | 证据（`results/raw/kaggle-1.5b-*`）|
| --- | --- | --- |
| 1 | 单请求 **vLLM 比 HF 快 2.45 倍**（1160.40 vs 2840.63 ms）；1→16 条 **吞吐 ×15.3**，延迟仅 +4.5% | `hf-single` / `vllm-single` / `vllm-batch16` |
| 2 | **`max_num_seqs` 的天花板 = 并发请求数**：16→64 ×1.42，之后 64→128 −1.4%、128→256 −2.0% | `concurrency-16/64/128/256` |
| 3 | 总 prefill 固定 8192 时延迟几乎不变（2346.75–2373.16 ms，**极差 1.13%**），**独立复现了 Part I 的 1.125%** | `context-512/2048/4096` |
| 4 | **16384 的代价在 T4 上是线性的（×2.10）**，而本机是次线性的（×1.50）| `context-8192/16384` |
| 5 | ⚠️ **prefix caching 异常**：shared-on 比 shared-off **慢 2.49 倍**，与 Part I 的 +21.9% **方向相反**，原因未解释 | `prefix-shared-{off,on}` |
| 6 | **chunked prefill 负收益（−48.3%）**，方向与本机一致但代价约 4 倍；本机那个 V1 参考结论在 T4 上无法验证 | `chunked-{off,on}` |
| 7 | **AWQ +13.3% 吞吐（延迟 ×0.883）**——本项目**第一个可归因的量化结论**（同引擎、同配额、同 prompt，唯一变量是权重位宽）| `awq` vs `vllm-batch16` |
| 8 | **TP=2 快 24.8%**（推翻 runbook 预设），**但每卡吞吐掉 37.6%** | `tp1` vs `tp2` |
| 9 | **噪声底 ±1.2%**：`tp1` 与 `concurrency-64` 是同一配置的两次运行（相隔 14 个 run），相差 1.19% | `tp1` vs `concurrency-64` |
| 10 | **E8 HTTP server 失败**（64/64 `APIConnectionError`），Kaggle 侧**零 per-request 指标** | `http-concurrency-16` |

### 记录缺陷（写结论时必须一并说明）

**Part I 暴露的三个：**

1. **`VLLM_USE_V1` 没有写进 JSONL**，导致本机 AWQ 臂事后无法确认跑在 V0 还是 V1 上——
   而 V0/V1 之间本身就有约 10% 的差距。
2. **除 server run 外没有 GPU 遥测**，E2/E3/E6 的显存全部无数据，"量化省了多少显存"无从回答。
3. **chunked prefill 只测了离线 batch**，而它的卖点是短请求尾延迟——设计上就观测不到。

**Part II 又暴露三个：**

4. **同一个 `VLLM_USE_V1` 缺陷在 Kaggle 上后果更重**：20 个 run 的"全部跑在 V0"这一事实
   **只来自操作者陈述，产物无法自证**——而 Part II 的量化与 TP 结论全都建立在它上面。
5. **Transformers 版本不可考，且环境快照与实际矛盾**：`env-kaggle-1.5b.json` 记的是 `5.0.0`，
   实际用的是降级后的 `4.51.1`；每次 run 的 metadata **根本不记 transformers**。该快照不可信。
6. **attention backend 无记录**，导致"16384 的代价为何两台机器不同"无法归因。

**其中 1 和 4 已在 2026-09-27 修补**：`scripts/common.py` 的 `environment_metadata()` 新增
`env_vars` 字段（记录 `VLLM_USE_V1`、`VLLM_WORKER_MULTIPROC_METHOD`、`VLLM_ATTENTION_BACKEND`、
`CUDA_VISIBLE_DEVICES`、`HF_HOME`）与 `transformers` 版本。**补丁只对未来的 run 生效**，
已有产物里的这些字段依然是缺的。

两个实测报错（V0 的 `max_num_batched_tokens ≥ max_model_len`、V1 不支持 fp8 KV cache）
已记录在 [`docs/runbook_three_models.md`](docs/runbook_three_models.md) 坑 6 / 坑 7；
Kaggle 撞到的六个环境问题记在报告第 13.2 节。


## 目录

```text
vllm-benchmark-lab/
├── README.md
├── project_summary.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/                     # 三个模型的 YAML 配置
├── scripts/
│   ├── common.py                # 共享 workload / 计时 / JSONL helper
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
    ├── runbook_three_models.md  # 三模型操作手册 + 坑 1–13（坑 8–13 是 Kaggle 专用）
    └── kaggle_runbook.md        # 早期 Kaggle 笔记（已归档为历史文档）
```

### 什么进版本库，什么不进

| 内容 | 是否提交 | 原因 |
| --- | --- | --- |
| `scripts/` `docs/` `configs/` `experiments/*/README.md` | ✅ | 源码与文档 |
| `results/*/` 目录结构（`.gitkeep`）| ✅ | 保证 clone 后目录完整 |
| `results/raw/*.jsonl`、`results/processed/*` | ❌ | 实验数据，本地生成 |
| `experiments/*/prompts_*.txt` + `.manifest.json` | ❌ | `make_prompts.py` 生成，seed=42 可精确重建 |
| `.claude/settings.local.json` | ❌ | 机器级权限白名单 |

**排除项都是可复现的生成物**，重建命令分别在：
`scripts/summarize_results.py`（聚合表）、`docs/runbook_three_models.md` 第 3.2 节（prompt 文件）。
因此本仓库 clone 下来是"能跑出同样结果"的状态，而不是"带着一堆产物"的状态。

## 常见限制

- 量化不是单纯的 dtype 开关，需要匹配实际量化 checkpoint。
- **T4 不等于本机 4060**；跨硬件结论必须分别标注。两台机器之间同时差着
  **GPU、模型、dtype、引擎**四个变量（报告 23.1），没有任何一对 run 是只差一个变量的。
- **Kaggle T4 只能跑 vLLM V0**（不支持 V1），且**没有 BF16、没有 FlashAttention-2、没有 Marlin kernel**。
  这四条硬件事实会同时改变结果的量级与归因方式。
- Tensor Parallel、两个单卡实例和单卡运行不是同一种实验，结果不能直接混排。
- **TP=2 的总吞吐与每卡吞吐是两个不同的问题**：Kaggle 实测 TP=2 总吞吐 +24.8%，
  但每卡均摊 **−37.6%**（报告 21.2）。
- OOM、未支持的 dtype 或缺少 checkpoint 都要作为失败实验保留，并写明原因。
  **跑失败的实验也要留文件**——`kaggle-1.5b-http-concurrency-16.jsonl`（64/64 连接失败）
  就是按这条规则保留下来的。
- **批级数字（离线 `LLM.generate`）与 per-request 数字不能混排**。
  Kaggle 侧**完全没有 per-request 数字**（server 实验失败），所以 T4 上的 TTFT/TPOT/p95
  一律不可得。

## 简历描述模板

> 构建可复现的 vLLM 推理基准实验室，统一比较 Transformers 与 vLLM 在不同模型、并发、上下文长度、prefix caching、chunked prefill 和量化策略下的 TTFT、TPOT、吞吐、显存和 GPU 利用率；支持本地单卡与 Kaggle T4 双卡实验，并通过 JSONL、聚合脚本和可视化报告完成结果追溯。
