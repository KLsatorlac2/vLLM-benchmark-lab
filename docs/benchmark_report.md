# Benchmark Report

本报告分两部分：

- **Part I（第 1–12 节）**：2026-09-25 / 09-26，本机 RTX 4060 上的 Qwen2.5-0.5B 全矩阵（E1–E6）。
  数字来自 `results/raw/local-0.5b-*.jsonl`。
- **Part II（第 13–23 节）**：2026-09-27，Kaggle T4 ×2 上的 Qwen2.5-1.5B 全矩阵（E1–E8）。
  数字来自 `results/raw/kaggle-1.5b-*.jsonl`。

两部分**口径独立**：不同 GPU、不同 dtype（4060 是 bfloat16，T4 只有 float16）、不同 tokenizer 之外的
一切都不共享。**跨部分直接比较数字是无效的**，第 23 节列出唯一允许的比较方式。

> **Part II 的关键前提**：T4 不支持 vLLM V1，所以 Kaggle 侧**除 HF 外全部 run 都跑在 V0 引擎上**
> （见 13.1）。这既让 Kaggle 内部的对照更干净，也让它与 Part I（本机默认 V1）之间多了一层引擎差异。

> **数据文件不在版本库中**（`.gitignore` 排除了 `results/raw/*` 与 `results/processed/*`，
> 只保留 `.gitkeep` 目录占位）。文中每一处数字都写明了它来自哪个 raw 文件，
> 在有数据的机器上可用 `scripts/summarize_results.py` 重建聚合表并逐条核对。
>
> ⚠️ **Part I 的 raw JSONL 在 2026-09-27 已不在本机 `results/raw/` 中**
> （该目录现在只有 `kaggle-1.5b-*.jsonl`）。**Part I 的数字仍可逐条核对**——
> `results/processed/summary-local-0.5b.md` 保留了全部 21 个 run 的 run 级数字
> 与 GPU 表——但**无法再从 raw 重新推导**。若需要重跑，命令见
> `docs/runbook_three_models.md` 第 4 节。

**Part I 只覆盖本机。** Kaggle 1.5B 已在 Part II 完成；Kaggle 7B 与"两个单卡实例"（E7b）仍未开始。
Part I 中凡是只有本机数据的结论，都不能外推到 T4。

## 1. Question

需要回答的核心问题：在相同模型、prompt、采样参数和输出长度下，HF `model.generate` 与 vLLM 的延迟、吞吐和显存表现有何差异？不同调度和缓存策略在哪些 workload 下有效？

## 2. Environment

| Field | Value |
| --- | --- |
| Execution environment | **local**（本机 WSL2）|
| Hostname | LAPTOP-P7FH055L |
| GPU model and count | NVIDIA GeForce RTX 4060 Laptop GPU ×1（8188 MiB，Ada / SM 8.9）|
| Driver / CUDA | CUDA 12.4 |
| Python / PyTorch / vLLM | Python 3.12.14 / torch 2.6.0+cu124 / vLLM 0.8.5.post1 |
| Transformers | 4.51.1 |
| Compute dtype | `dtype=auto` → **bfloat16**（Ada 支持 BF16）|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`，revision 未固定（`revision=null`，用 HF cache 默认）|
| Quantized checkpoints | `Qwen/Qwen2.5-0.5B-Instruct-AWQ`（E6）|
| Parallel mode | **single GPU**，`tensor_parallel_size=1` |
| Virtualenv | `/home/asus/venvs/vllm` |
| Kaggle T4 ×2 | **1.5B 已完成**，见 Part II（第 13–23 节）；7B / 双实例未执行 |

环境字段来自每次 run 的 metadata 行（`environment_metadata()`），不是手工填的。

### 2.1 引擎版本（关键，必须逐臂标注）

本机默认跑 **V1** 引擎。第 9、10 节的三个臂因为功能限制改用了 V0：

| 臂 | 引擎 | 原因 |
| --- | --- | --- |
| E1–E4 全部、E6 的 AWQ | V1（默认）| — |
| E5 的 `off` / `on` | **V0**（`VLLM_USE_V1=0`）| V1 下 chunked prefill 恒开，关不掉 |
| E5 的 `v1ref` | V1 | 作为 V1 默认配置的参考 |
| E6 的 `kvfp8` | **V0**（`VLLM_USE_V1=0`）| V1 未实现 fp8 KV cache，会抛 `NotImplementedError` |

> **已知缺陷**：`VLLM_USE_V1` **没有写进 JSONL**，所以 `local-0.5b-awq.jsonl` 的引擎版本
> 无法从产物中确认（见表注 10.2）。上表中除 kvfp8 外，其余按"本机默认 V1"推定。
> 详见 `docs/runbook_three_models.md` 坑 7。

## 3. Workload

所有 prompt 由 `scripts/make_prompts.py` 生成，**精确 token 长度**并逐条复核，
workload 分布记在生成的 `experiments/*/prompts_*.txt.manifest.json`。

> **prompt 文件（`experiments/*/prompts_*.txt` 与同名 `.manifest.json`）不在版本库中**
> ——它们是 `make_prompts.py` 生成的数据，`.gitignore` 已排除。
> 重建命令与固定 seed（42）见 `docs/runbook_three_models.md` 第 3.2 节，可逐字节复现。
> 下表就是按那些命令的参数整理的。

| 实验 | prompt 文件 | 条数 | 单条 token | 总 prefill token | 输出上限 |
| --- | --- | --- | --- | --- | --- |
| E1 基线 | `common.py` 默认 prompt | 1 / 16 | 10–12 | 10（单请求）/ 176（16 请求）| 64 |
| E2 并发 | `concurrency/prompts_512x64.txt` | 64 | 512（互不相同）| 32768 | 64 |
| E3 上下文 | `context_length/prompts_{512,2048,4096,8192}.txt` | 16 / 4 / 2 / 1 | 512 / 2048 / 4096 / 8192 | **固定 8192** | 64 |
| E3 上下文（加长）| `context_length/prompts_16384.txt` | 1 | 16384 | 16384 | 64 |
| E4 prefix | `prefix_caching/prompts_{shared,random}_1024.txt` | 16 | 1152（1024 前缀 + 尾部）| 18432 | 64 |
| E5 chunked | `chunked_prefill/prompts_mixed.txt` | 16 | 3584 ×8 / 64 ×8 | 29184 | 64 |
| E6 量化 | `common.py` 默认 prompt | 16 | 10–12 | 176 | 64 |

- Prompt source：程序生成，非真实语料；`distinct` / `shared-prefix` / `random-prefix` / `mixed` 四种模式。
- Sampling：`temperature=0.0`（贪心），`seed=42`，确定性。
- Warmup policy：**离线 `LLM.generate` 没有排除 warmup**——`torch.compile` 的编译耗时（首次约 51 s）发生在
  第一次 `generate` 之内，会污染第一个 batch。本报告的离线数字因此偏向保守；server 模式有 2 条 warmup 请求并已排除。
- Requests 与 repeats：每个配置 **只跑 1 次**，没有重复实验，所以**无法给出运行间方差**。见第 12 节。

### 3.1 E3 为什么固定总 prefill token

512×16、2048×4、4096×2、8192×1 四组的**总 prefill 工作量都是 8192 token**。
这样 batch 延迟的变化只来自"单条序列变长"，而不是"总工作量变多"——否则长度和总量两个变量会混在一起。

代价是 **batch 宽度同时从 16 降到 1**，而 batch 宽度决定 decode 的并行度。
所以 E3 的吞吐差异里混了长度和 batch 宽度两件事，第 7 节会把这个分开说清楚。

## 4. Metrics

- `TTFT`、`TPOT`、end-to-end latency：**只有 server 模式（`vllm_server`）提供 per-request 值**。
- p50/p95/p99：同上；离线 batch run 的百分位被 `summarize_results.py` 自动隐藏。
- `output_tok_per_s`：离线 run 是 **batch 吞吐**（整个 batch 的输出 token / 整个 batch 的耗时）。
- `req/s`：只有 server 模式记录。
- GPU utilization / memory / power / temperature：来自 `nvidia-smi` 时间序列（`collect_gpu_metrics.py`）。
- **KV cache usage：全部 run 都记作 `unavailable`**。离线 runner 没有导出 vLLM 的 KV cache 命中/占用指标，
  本机也没解析引擎日志，因此本报告**不提供任何 KV cache 占用数字**，也不拿总显存占用替代。

### 4.1 两种口径不能混排

| `latency_mode` | 来源 | 能讲什么 |
| --- | --- | --- |
| `batch` | `run_vllm.py` 离线 | batch 延迟、batch 吞吐、批处理收益 |
| `single` | 同上，n=1 | 单请求延迟（batch 与 per-request 重合）|
| `per-request` | `benchmark_http.py` server | TTFT / TPOT / p95 / p99 / req/s |

## 5. HF vs vLLM

E1 拆成两步，把"引擎差异"和"批处理差异"分开。

### 5.1 E1a 单请求对单请求（纯引擎差异）

同一模型、同一 prompt（10 token）、`max_tokens=64`、`seed=42`：

| run | backend | batch 延迟 | output tok/s |
| --- | --- | --- | --- |
| `local-0.5b-hf-single` | HF `model.generate`（逐条串行）| **3675.41 ms** | **17.41** |
| `local-0.5b-vllm-single` | vLLM 离线，`max_num_seqs=1` | **986.91 ms** | **64.85** |

**vLLM 单请求延迟是 HF 的 0.269 倍（快 3.7×），吞吐 3.72 倍。**
HF 的 `tpot_ms=57.43`（≈ 3675.41/64，按全部输出 token 摊），vLLM 单请求约 15.4 ms/token。

这一步是干净的引擎对比：两边都是 batch=1，差异全部来自引擎实现
（HF 逐 token Python 循环 + 无 PagedAttention / 无 CUDA graph；vLLM 有连续批处理调度和编译后的执行图）。

### 5.2 E1b 批处理收益（vLLM 自己 1 → 16）

| run | requests | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- |
| `local-0.5b-vllm-single` | 1 | 986.91 ms | 64.85 tok/s |
| `local-0.5b-vllm-batch16` | 16 | **952.11 ms** | **1075.51 tok/s** |

**吞吐提升 16.6 倍，而 batch 延迟几乎不变（×0.965，反而略降）。**

这是本轮最干净的一条结论：在 0.5B 这个规模上，**16 条并发请求的显存和算力开销还没有把 GPU 压满**，
批处理几乎是把空闲的算力直接换成了吞吐。注意 16 条请求的输出总量是单条的 16 倍，
却用了**更少**的总时间——说明单请求时 GPU 严重未饱和。

### 5.3 与 server 模式的交叉验证

server 模式（`local-0.5b-http-concurrency-16`，并发 16，同一份 512×64 workload）
吞吐 **941.23 tok/s**，离线 `concurrency-16` 是 **1005.55 tok/s**，比值 **0.936**。

两条完全独立的代码路径（离线 `LLM.generate` vs HTTP + OpenAI 协议）在 6% 以内吻合，
**互相验证了对方没有系统性错误**。server 低 6% 可以归因于 HTTP/JSON 序列化开销，以及两边引擎配置的差异。

## 6. Concurrency

E2：64 个请求（512 token prompt，互不相同），只改 `max_num_seqs`，`gpu_memory_utilization=0.75`。

| `max_num_seqs` | batch 延迟 | batch 吞吐 | 每请求均摊 | 相对上一档吞吐 |
| --- | --- | --- | --- | --- |
| 16 | 4073.41 ms | 1005.55 tok/s | 63.65 ms | — |
| 32 | 3015.13 ms | 1358.48 tok/s | 47.11 ms | **×1.351** |
| 64 | **2364.97 ms** | **1731.95 tok/s** | **36.95 ms** | **×1.275** |

观察点：

1. **吞吐随 batch 宽度上升，但边际收益递减**：16→32 是 1.35 倍，32→64 只有 1.275 倍。
   到 64 条时 GPU 开始接近饱和，加宽 batch 换来的收益在变小。
2. **batch 延迟随宽度上升而下降**，因为 64 个请求从"4 批 × 16"变成"1 批 × 64"，总排队轮次减少。
   端到端 **4073 → 2365 ms（×0.58）**。
3. **没有观测到 OOM**。`max_num_seqs=64`、512 token 上下文、`gpu_memory_utilization=0.75` 在 8GB 上跑得下。
   本轮**没有继续向上扫**（128/256），所以**没有找到显存上限点**——8GB 上的 OOM 边界仍是未知。
4. **本轮没有 GPU 采集器伴随运行**，所以 E2 各档的显存占用**没有数据**，不能给出"哪一档接近显存上限"的判断。
   唯一的显存数字来自 4.7 节的 server run（见 6.1）。

### 6.1 唯一的 GPU 遥测（伴随 server run）

`results/raw/gpu-local-0.5b-http-16.jsonl`，`gpu-memory-utilization=0.6`、120 s、0.5 s 间隔、104 个采样点：

| GPU | 峰值显存 | 显存上限 | 峰值占比 | 平均利用率 | 最大利用率 | 最大功耗 | 平均温度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 4060 Laptop | 4382.0 MB | 8188.0 MB | **53.5%** | 6.3% | **85.0%** | 54.0 W | 52.5 °C |

> **平均利用率 6.3% 不能读成"推理时 GPU 空闲"。** 采集窗口 120 s 覆盖了模型加载、
> `torch.compile` 和一大段空闲等待，真正的压测窗口只有约 4.35 s（`wall_ms=4351.74`），
> 被稀释掉了。**要看的是最大利用率 85%**，它说明压测期间 GPU 确实被用起来了。
>
> **峰值显存 53.5% 是在 `gpu-memory-utilization=0.6` 下测的**，不代表 0.75/0.90 配置的占用。
> 两个数字都不该外推到 E2/E6 的配置上。

## 7. Context Length

E3：总 prefill 固定在 8192 token（前四行），只改单条长度与 batch 宽度。

| 单条长度 | batch 宽度 | 总 prefill | batch 延迟 | batch 吞吐 | `max_model_len` |
| --- | --- | --- | --- | --- | --- |
| 512 | 16 | 8192 | 1373.90 ms | 745.32 tok/s | 1024 |
| 2048 | 4 | 8192 | 1404.20 ms | 182.31 tok/s | 2560 |
| 4096 | 2 | 8192 | 1545.25 ms | 82.83 tok/s | 4608 |
| 8192 | 1 | 8192 | 1475.21 ms | 43.38 tok/s | 8704 |
| **16384** | 1 | **16384** | **2212.62 ms** | **28.92 tok/s** | 16896 |

**实际 `input_tokens` 与上表"单条长度"逐条核对一致**（512/2048/4096/8192/16384），不是配置上限。

### 7.1 结论一：总工作量不变时，延迟与单条长度基本无关

前四行的延迟落在 **1373.90 – 1545.25 ms**，最大/最小只有 **1.125 倍**，
而且**不单调**（4096 那组反而比 8192 高）。也就是说在 8192 token 的总量下，
"16 条 × 512" 和 "1 条 × 8192" 的耗时是同一量级。

**prefill 的代价大致正比于总 token 数，而不是单条序列的长度**——这是 PagedAttention + 分块调度的直接结果。

### 7.2 结论二：吞吐的暴跌是 batch 宽度效应，不是长上下文效应

吞吐从 745.32 掉到 43.38 tok/s（**×0.058**），看起来像"长上下文很慢"，但**不是**：

- 总 prefill 都是 8192 token，延迟几乎没变（7.1）；
- 变的是 **batch 宽度 16 → 1**，也就是 decode 的并行路数从 16 条降到 1 条；
- 每组的输出总量 = 64 × batch 宽度，从 1024 token 降到 64 token。

同样的 64 个输出 token，16 条序列并行 decode 和 1 条序列串行 decode，差距本来就在 16 倍量级。
**所以这张表不能读成"上下文越长吞吐越低"，只能读成"batch 越窄吞吐越低"。**
E2 的 `max_num_seqs` 扫描是同一个效应（第 6 节）。

### 7.3 结论三：16384 的代价

16384 那组是**同样的 batch 宽度 1、但 prefill 工作量翻倍**，可以和 8192 那组直接比：

**1475.21 → 2212.62 ms，×1.50**。2 倍 token 换来 1.5 倍时间，**慢于线性但同量级**，
说明在本机 0.5B + 16K 长度下还没有出现明显的注意力平方级爆炸。

0.5B 的 KV cache 约 12 KB/token，16384 只需约 196 MB，8GB 显存毫无压力——
**这是本机相对 T4 的独特优势**（T4 上 7B 做 16384 要 917 MB）。

## 8. Prefix Caching

E4：两个 workload 各跑 `--enable-prefix-caching` 开关两臂，`max_num_seqs=16`、`max_model_len=1280`、16 条 × 1152 token。

| workload | caching | batch 延迟 | batch 吞吐 | 相对 off |
| --- | --- | --- | --- | --- |
| **shared**（1024 共享前缀 + 尾部）| off | 1613.21 ms | 634.76 tok/s | — |
| | **on** | **1323.59 ms** | **773.65 tok/s** | **延迟 ×0.820，吞吐 +21.9%** |
| **random**（1024 各自唯一前缀 + 尾部）| off | 1579.45 ms | 648.33 tok/s | — |
| | on | 1597.91 ms | 640.84 tok/s | 延迟 ×1.012，吞吐 **−1.2%** |

### 8.1 结论

1. **前缀真的共享时，缓存有明确收益：吞吐 +21.9%，延迟 −18%。**
2. **前缀不共享时，开关缓存没有任何收益（−1.2%，在噪声范围内）。**
   这正是预期行为——`random` 臂是 `shared` 臂的对照组，用来证明收益确实来自前缀复用，
   而不是"打开某个开关就变快"。
3. **两个 off 臂几乎相同（1613.21 vs 1579.45 ms，相差 2%），说明两个 workload 除"前缀是否共享"外是对等的**，
   归因可以干净地落在前缀共享这一点上。
4. **收益（+21.9%）远小于可缓存比例（1024/1152 = 88.9%）**。原因是 prefill 只是总工作量的一部分，
   16 条请求各自还要 decode 64 个 token，这部分不受 prefix caching 影响。
   所以"缓存能省掉 89% 的 prefill"不等于"吞吐提升 89%"。

> 口径提醒：本节是**离线 batch 吞吐**，不是 per-request TTFT。
> prefix caching 对**单请求 TTFT** 的影响需要用 server 模式测，本轮未做。

## 9. Chunked Prefill

E5：`prompts_mixed.txt`（8 条 3584 token + 8 条 64 token 交替），`max_model_len=4096`、`max_tokens=64`、16 请求。
**OFF 和 ON 两臂跑在 V0 上**，V1 参考臂跑在 V1 上（原因见 2.1）。

| 臂 | 引擎 | `max_num_batched_tokens` | 切块 | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- | --- | --- |
| `off` | V0 | 4096 | 否 | **2196.55 ms** | **466.19 tok/s** |
| `on` | V0 | 2048 | 是 | 2487.99 ms | 411.58 tok/s |
| `v1ref` | V1 | 2048 | 恒开 | 2237.98 ms | 457.56 tok/s |

### 9.1 结论一：离线 batch 下，开切块反而更慢（−11.7%）

`on` 比 `off` 慢 11.7%。这与"chunked prefill 提升性能"的直觉相反，但在**离线 batch 模式**下是合理的：

- 离线模式**所有 16 个请求同时入队**，本来就是一个大 batch，不存在"短请求被长请求堵住"的场景；
- 切块把长 prefill 拆成 2 步，**多了调度开销和额外的中间状态管理**，却换不到"短请求插队"的收益。

**切块的收益是尾延迟收益，不是吞吐收益。** 而离线 `LLM.generate` 只测得到整体耗时，
**测不到短请求的 TTFT**——所以这个实验设计本身就观察不到它的卖点。见 9.3。

### 9.2 结论二：V1 的切块实现比 V0 快

`v1ref`（V1，切块，预算 2048）比 `on`（V0，切块，预算 2048）**快 11.2%**，
和 `off`（V0，不切块，预算 4096）**几乎持平（−1.9%）**。

`on` 和 `v1ref` 的切块配置相同、只差引擎版本，所以这 11.2% 可以较干净地归因给**引擎实现**：
**V0 的 chunked prefill 有额外开销，V1 把它做得更好**。这与"V1 下 chunked prefill 恒开且是默认路径"的设计一致。

### 9.3 这个实验没有回答的问题

**"切块是否改善短请求的尾延迟"——本轮没有测到。**
需要 server 模式 + per-request TTFT（操作见 `docs/runbook_three_models.md` 4.7 节）才能观测，本轮未执行。
因此本节只能给出"离线吞吐"的结论，**不能声称 chunked prefill 无效**。

> **配置口径提醒**：两臂的 `max_num_batched_tokens` 不同（4096 vs 2048）是**必然的**，不是污染。
> 关掉切块时单条 prompt 必须能在一个 step 里放下，所以预算必须 ≥ 3584。
> 这个差异本身就是"切块 vs 不切块"的操作定义。另外三个臂的 `max_model_len` 统一为 **4096**
> （原设计的 4608 在 V0 下会启动即崩，见 `runbook_three_models.md` 坑 6）。
> 最长 prompt 只有 3584，**实际用到的长度没有变**，三臂依然可比。

## 10. Quantization

E6：同一份默认 prompt（10–12 token）、16 请求、`max_tokens=64`。

### 10.1 结果

| 臂 | checkpoint / 设置 | 引擎 | `gpu_mem_util` | batch 延迟 | batch 吞吐 | 相对 FP16 基线 |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 基线 | `Qwen2.5-0.5B-Instruct` | V1 | 0.90 | 952.11 ms | 1075.51 tok/s | — |
| AWQ | `Qwen2.5-0.5B-Instruct-AWQ` | 未记录（推定 V1）| 0.75 | 1097.54 ms | 932.99 tok/s | **−13.3%** |
| FP8 KV cache | `kv_cache_dtype=fp8` | **V0** | 0.75 | 1094.47 ms | 935.61 tok/s | **−13.0%** |

**这三条数字不能直接互比**，原因见 10.2。GPTQ 本轮**未执行**（checkpoint 未验证/未跑）。
量化臂的**显存占用没有采集**，所以本报告**不提供任何"量化省了多少显存"的数字**——
项目规则要求 KV cache usage 只在 vLLM 明确提供时才记录，本轮不可得。

### 10.2 为什么不能互比（三个混淆变量）

1. **引擎版本不同**：`kvfp8` 确定跑在 V0 上（V1 不支持 fp8 KV cache），
   基线跑在 V1 上，`awq` 的引擎版本**无法从 JSONL 确认**（`VLLM_USE_V1` 没被记录，见 2.1）。
   E5 已经证明 **V0 与 V1 之间就有约 10% 的差距**（9.2），和这里的 13% 是同一量级。
2. **`gpu_memory_utilization` 不同**：基线 0.90，量化臂 0.75。
3. **prompt 不同**：量化臂用的是默认 prompt（10 token）；`runbook` 里建议的"同参数基线"
   `concurrency-16` 用的是 512 token prompt，**两者差 50 倍，batch 延迟根本不可比**。
   本节改用 `vllm-batch16`（同一份默认 prompt）作基线，但它仍有变量 1 和 2。

**因此本节只能写"量化臂在本机 0.5B 上没有可观测的吞吐收益"，不能写"fp8/AWQ 比 FP16 慢 13%。"**

### 10.3 输出 sanity check

AWQ 与 fp8 KV 两臂都用 `--include-text` 保存了生成文本，与同一份 prompt 的 FP16 输出逐条比对：

- 两臂输出**都是连贯英文**，正确回答了 4 个 prompt（batching / TTFT vs TPOT / 上下文窗口 / KV cache），
  **没有乱码、没有复读崩溃**。
- 两臂文本**互不相同**（16/16 条都不同）。这是**预期行为**，不是错误：
  AWQ 改的是权重，fp8 改的是 KV cache 精度，两者都会改变数值轨迹。
- `kvfp8` 的 `tpot`/复读表现：有一条输出出现了明显的重复句式（"I have a question about..." 连写数遍），
  在贪心解码下这是**质量退化的信号**，但样本量只有 1 条，**不足以支撑"fp8 KV 降低输出质量"的结论**，
  只能作为后续需要更大规模评测（如困惑度或任务准确率）的理由。

### 10.4 未执行的量化臂

| 组合 | 状态 |
| --- | --- |
| GPTQ | **未执行**（本机未验证 checkpoint）|
| AWQ / GPTQ / fp8 KV on T4 | **未执行**（Kaggle 部分未开始）|
| 量化臂的显存对比 | **无数据**（未采 GPU 遥测）|

> 按项目规则，未执行的组合标为"未执行"，**不写成失败实验**——失败实验指跑过并崩掉的配置。
> 本轮的失败实验是两个崩溃配置（4608 的 E5 OFF 臂、V1 上的 fp8 KV），
> 它们在写 JSONL 之前就退出了，**没有留下结果文件**，只在
> `docs/runbook_three_models.md` 坑 6 / 坑 7 里留了记录。

## 11. Conclusions

以下每条都能回溯到 `results/raw/` 的具体文件。

### 11.1 低并发场景的延迟选择

**用 vLLM，而不是 HF。** 单请求下 vLLM 延迟 986.91 ms vs HF 3675.41 ms（**3.7 倍**），
吞吐 64.85 vs 17.41 tok/s。这个差距在 batch=1 时测出，**纯粹来自引擎实现**，与批处理无关（5.1）。

**长上下文不构成延迟障碍。** 总 prefill 8192 token 时，无论切成 16 条 512 还是一条 8192，
延迟都在 1374–1545 ms（7.1）。单条 16384 只比 8192 慢 1.5 倍（7.3）。

### 11.2 高并发场景的吞吐选择

**把 `max_num_seqs` 开大，直到显存或边际收益不允许。** 64 个 512-token 请求：

- `max_num_seqs=16` → 1005.55 tok/s，端到端 4073 ms
- `max_num_seqs=64` → 1731.95 tok/s，端到端 2365 ms（**吞吐 ×1.72，延迟 ×0.58**）

**边际收益在递减**（16→32 是 ×1.351，32→64 只有 ×1.275），说明接近饱和。
**8GB 上 64 并发没有 OOM**，但**本轮没找到显存上限点**。

**批处理本身是最大的一笔收益**：vLLM 自己从 1 条加到 16 条，吞吐 ×16.6，而 batch 延迟几乎不变（5.2）。

### 11.3 长上下文场景的显存限制

**本机 0.5B 没有触到任何显存限制。** 16384 上下文只占约 196 MB KV cache，
唯一一次 GPU 遥测的峰值显存是 4382/8188 MB = **53.5%**（且是在 `gpu-memory-utilization=0.6` 下）。

**真正的显存限制在本轮未被探测到**——E2 没有向上扫到 OOM，也没有伴随显存采集。
本机 8GB 的边界、以及 T4 16GB 上 1.5B/7B 的边界，都还是空白。

### 11.4 缓存策略的选择

**prefix caching 只在真有共享前缀时有用。** shared workload 开缓存 **+21.9% 吞吐**，
random workload 开缓存 **−1.2%**（8.1）。这条对照是本节最重要的设计：
**没有 random 对照组，就不能排除"开关本身带来收益"的解释。**

注意收益幅度（+21.9%）远低于可缓存比例（88.9%），因为 decode 部分不受影响。

**chunked prefill 在离线 batch 下是负收益（−11.7%）**，但这不是它的使用场景（9.1）。
它的卖点是尾延迟，**本轮设计测不到**，结论应记为"未验证"而非"无效"。
顺带测到一条实现层面的结论：**V1 的切块比 V0 快 11.2%**（9.2）。

### 11.5 量化带来的收益与代价

**本轮没有测到量化的速度收益，也没有测到显存收益（没采显存）。**
AWQ 和 fp8 KV 相对 FP16 基线都是 −13% 吞吐，但**这个 13% 归因不了**
——引擎版本（V0/V1 差约 10%）和 `gpu_memory_utilization` 都没对齐（10.2）。

**输出质量**：两臂文本都连贯可读，但与 FP16 逐条不同（预期）。fp8 KV 有一条出现重复句式，
**样本量不足以定性**（10.3）。

按项目规则，这一节的正确写法是：**"量化在本机 0.5B 上是资源受限场景的工具，
本轮未能给出干净的速度/显存结论；AWQ 可加载、fp8 KV 需退回 V0、GPTQ 未验证。"**

### 11.6 不可横向比较的限制（必须随结论一起引用）

1. **本机 4060 与 Kaggle T4 不可直接比较**：不同 GPU，且 dtype 不同（4060 是 bfloat16，T4 只有 float16）。
   写 Part I 时 Kaggle 部分尚未执行；**Kaggle 1.5B 已于 2026-09-27 完成（Part II）**，
   但跨部分比较又多出"模型 0.5B vs 1.5B"和"引擎 V1 vs V0"两个变量，
   **四个变量同时不同，绝对数字依然不可比**。完整清单见第 23.1 节。
2. **离线 batch 数字与 server per-request 数字不能混在同一张表里比**。
   本报告已分开（第 5–10 节是 batch，5.3 / 6.1 是 server）。
3. **TP=2、两个单卡实例、单卡运行是三种不同实验**。本机只有 1 张卡，**E7 跳过**；
   Kaggle 上 TP=1 vs TP=2 已测（Part II 第 21 节），**两个单卡实例（E7b）仍未执行**。
4. **chunked prefill 的结论限于 V0**；V1 下该特性恒开，不能关闭。
5. **每条结论只对应一个 run，没有重复实验**，见第 12 节。
   （Part II 已通过一对意外重复测出噪声底 ±1.2%，见第 13.5 节。）

## 12. 本轮的局限（写结论时应一并说明）

| # | 局限 | 影响 |
| --- | --- | --- |
| 1 | **每个配置只跑 1 次，无重复** | 报不出运行间方差；±10% 以内的差异（如 9.2 的 11.2%）严格来说不能排除机器噪声 |
| 2 | **`VLLM_USE_V1` 未写入 JSONL** | `awq` 臂的引擎版本无法事后确认，E6 的归因 permanently 受损 |
| 3 | **离线无 warmup 排除** | `torch.compile`（约 51 s）落在首次 `generate` 内，离线数字偏保守 |
| 4 | **除 server run 外没有 GPU 遥测** | E2/E3/E6 的显存、利用率、功耗全部无数据；量化"省显存"无从验证 |
| 5 | **无 KV cache usage 指标** | 全程 `unavailable`，符合项目规则（不用总显存替代）|
| 6 | **chunked prefill 未测 server 模式** | 尾延迟结论缺失，该特性实际上未被真正评估 |
| 7 | **量化基线未对齐** | E6 的三个臂在引擎、显存配额、prompt 上都不一致 |
| 8 | ~~**Kaggle 1.5B / 7B / TP 全部未执行**~~ **Kaggle 1.5B 已于 2026-09-27 完成**（Part II）；7B 与双实例仍未执行 | 三模型矩阵完成 2/3；跨硬件结论仍受 23.1 的引擎/dtype 差异限制 |
| 9 | **E2 未扫到 OOM** | 8GB 的 `max_num_seqs` 上限未知 |
| 10 | **prompt 为程序生成** | 非真实语料，绝对吞吐不能外推到生产流量 |

优先级最高的两条是 **#2**（补记录，否则已有结果无法归因）和 **#4/#6**（补遥测和 server 测量，
否则量化和 chunked prefill 两节实际上没有有效结论）。待办清单见
`docs/runbook_three_models.md` 第 10 节。

> **#2 已于 2026-09-27 修补**（`scripts/common.py` 的 `environment_metadata()` 新增
> `env_vars` 与 `transformers` 字段），**但只对未来的 run 生效**。
> **#4/#6 在 Part II 里依然没有解决**——Kaggle 侧的遥测和 server 模式都没拿到，
> 见第 22 节与第 23.2 节。

---

# Part II — Kaggle T4 ×2 / Qwen2.5-1.5B（2026-09-27）

覆盖 E1–E7，外加一次失败的 E8（HTTP server）。共 **22 个产物文件**：
20 个 vLLM 离线 run + 1 个 HF run + 1 个失败的 server run，全部在 `results/raw/kaggle-1.5b-*.jsonl`。

聚合表由 `scripts/summarize_results.py` 重建（命令见 `docs/runbook_three_models.md` 7.1）：

```bash
python scripts/summarize_results.py results/raw/kaggle-1.5b-*.jsonl \
  --output-md results/processed/summary-kaggle-1.5b.md \
  --output-csv results/processed/summary-kaggle-1.5b.csv
```

## 13. 环境与六个已知问题

### 13.1 环境

| Field | Value |
| --- | --- |
| Execution environment | **Kaggle Notebook**（T4 ×2）|
| Hostname | `6b937f1eb901`（容器名；22 个 run 全部相同 → 同一次 session）|
| GPU model and count | Tesla T4 ×2（各 16 GB，Turing / **SM 7.5**）|
| Driver / CUDA | CUDA 12.4 |
| Python / PyTorch / vLLM | Python 3.12.13 / torch 2.6.0+cu124 / vLLM 0.8.5.post1 |
| Transformers | **未记录**（见 13.4）|
| Compute dtype | `dtype=half`（T4 无 BF16，见 13.2 问题 4）**全部 20 个 vLLM run 一致** |
| Attention backend | **XFormers**（T4 上 FA2 不适用，见 13.2 问题 5）|
| Engine | **V0**（T4 不支持 V1，见 13.2 问题 1，展开在 13.3）|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`；E6 另一臂为 `Qwen/Qwen2.5-1.5B-Instruct-AWQ` |
| Parallel mode | 单卡（`CUDA_VISIBLE_DEVICES=0`）；仅 `kaggle-1.5b-tp2` 用双卡 |
| GPU 遥测 | **无**。Kaggle 侧一个 `gpu-*.jsonl` 都没有采集 |
| `gpu_memory_utilization` | 0.85（全部 22 个 run 一致）|

上表除 Transformers 一行外，其余字段都来自每次 run 的 metadata 行，可逐条核对。

### 13.2 六个已知问题总表

Kaggle 环境与 runbook 预设不符，实际撞到六个问题。**问题 4 和 6 能从产物自证，问题 1/2/3/5 不能**
——这也是 13.4 那三个记录缺陷的来源。

| # | 问题 | 现象 | 处理 | 对结果的影响 |
| --- | --- | --- | --- | --- |
| 1 | **T4 不支持 vLLM V1** | V1 在 SM 7.5 上不可用 | 除 HF 外**全部退回 V0**（`VLLM_USE_V1=0`）| 见 13.3，影响最大 |
| 2 | **Transformers 5.0.0 不兼容** | 与 vLLM 0.8.5.post1 冲突 | 从 5.0.0 **降到 4.51.1** | 见 13.4 缺陷 2 |
| 3 | **TensorFlow / protobuf 冲突** | Kaggle 预装 TF 与 vLLM 的 protobuf 要求冲突 | **卸载 TensorFlow**（实验不需要）| 无（不参与推理），但**未记录**，下个 session 会重现 |
| 4 | **T4 不支持 BF16** | Turing 无 BF16 | 所有 vLLM 命令加 `--dtype half` | ✅ 可自证：20/20 个 run 的 `dtype` 都是 `half`；与本机 `bfloat16` 不同，是跨硬件比较的第二个混淆变量 |
| 5 | **T4 无 FlashAttention-2** | 日志报 FA2 不适用于该 GPU | vLLM 用 **XFormers** backend | 长上下文性能，见第 17 节 |
| 6 | **Kaggle 不允许后台进程** | `nohup ... &` 起的 server 随 cell 结束被回收 | HTTP server 实验**未完成** | 见第 22 节（失败实验）|

> **问题 1 与问题 5 很可能同源**：T4 是 SM 7.5，而 V1 依赖 FlashAttention 系 kernel（需要 SM 8.0+）。
> 也就是说"V1 不可用"和"FA2 不可用"是同一个硬件事实的两个侧面。
> **这是推断，不是实测**——本次没有保存 vLLM 启动日志，无法从产物确认。

### 13.3 问题 1 展开：Kaggle 全部跑在 V0 上

**这一条决定了 Part II 全部结论的可比性边界，四条影响：**

**(a) Kaggle 内部的对照因此更干净。** 20 个 vLLM run 全在同一引擎（V0）上，
所以第 20 节的 AWQ vs FP16、第 21 节的 TP=1 vs TP=2 **都是同引擎对照**。
这正是 Part I 的 E6 缺的东西——Part I 第 10.2 节里三个臂的引擎、显存配额、prompt 全不一致，
13% 的差**归因不了**。**Kaggle 的量化一节回答了本机答不了的问题。**

**(b) 但与 Part I 的跨硬件比较多了一层混淆。** 本机默认 V1，且 Part I 第 9.2 节实测
**V0 与 V1 之间本身就有约 10% 的差距**。所以 Part I vs Part II 的任何数字差里，
除了"GPU 不同 + 模型不同（0.5B vs 1.5B）+ dtype 不同（bf16 vs fp16）"，
还要再加一条"**引擎不同**"。四个混淆变量叠在一起，**跨部分比较数字不可行**（见第 23 节）。

**(c) E5 的 V1 参考臂在 T4 上无法执行。** Part I 有一条结论"V1 的切块实现比 V0 快 11.2%"
（Part I 第 9.2 节），它靠 `v1ref` 臂支撑。Kaggle 上 V1 不可用，**这一条无法在 T4 上交叉验证**。

**(d) Part I 的坑 1、坑 2 在 Kaggle 上不适用。** 那两条（V1 下 chunked prefill 关不掉、
V1 下 prefix caching 默认开）都是 **V1 行为**。V0 下 chunked prefill 可以真的关掉、
prefix caching 默认就是关的。所以 Kaggle 的 `--enable-prefix-caching` 开关语义**比本机更直接**
——off 臂不传参就是真的 off，不存在"省略参数静默变成开"的陷阱。
**这让第 18 节那个异常更费解**：按理说 Kaggle 的 E4 设计比本机更干净。

> **记录缺陷（最严重的一条）**：`environment_metadata()` 原先**不记 `VLLM_USE_V1`**，
> 所以"Kaggle 全部跑在 V0"这一事实**只来自操作者陈述，产物无法自证**。
> 已于 2026-09-27 补上（`scripts/common.py` 新增 `env_vars` 字段），但**对本文的 22 个文件无效**
> ——它们都是补丁之前生成的。

### 13.4 三个记录缺陷

| # | 缺陷 | 后果 |
| --- | --- | --- |
| 1 | **`VLLM_USE_V1` 未写入 JSONL** | 本文所有 run 的引擎版本无法从产物确认，只能按操作者陈述记为 V0。Part I 第 12 节 #2 是同一个缺陷 |
| 2 | **Transformers 版本未写入 metadata，且环境快照与实际不符** | `results/raw/env-kaggle-1.5b.json` 记的是 `transformers: 5.0.0`，而实际使用的是降级后的 **4.51.1**。该快照首行的日志时间戳是 `09-27 07:18:41`。每个 run 的 metadata 行**根本不记 transformers**，所以哪个 run 用哪个版本**不可考** |
| 3 | **Attention backend / FA2 不可用未写入 JSONL** | 第 17 节要解释"16384 的代价为何与 Part I 不同"，backend 是关键变量，却无记录 |

> **结论：`env-kaggle-1.5b.json` 不可信，需重拍。**
> 重拍前先确认 transformers 版本，并让 `run_vllm.py` 把实际解析出的 attention backend 也写进 metadata。

### 13.5 意外获得的重复实验：噪声底 ±1.2%

`kaggle-1.5b-tp1`（pid 2434）与 `kaggle-1.5b-concurrency-64`（pid 1051）**配置完全相同**
（同模型、同 prompt 文件、`requests=64`、`max_num_seqs=64`、`max_model_len=1024`、
`gpu_memory_utilization=0.85`、`tensor_parallel_size=1`），是同一次 session 里跑了两次：

| run | pid | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- |
| `concurrency-64` | 1051 | 6519.28 ms | 628.29 tok/s |
| `tp1` | 2434 | 6596.69 ms | 620.92 tok/s |
| **相差** | — | **+1.19%** | **−1.19%** |

**这是本项目第一次拿到运行间方差，价值很高：**

1. **噪声底 ≈ ±1.2%。** 小于这个幅度的差异**不能解释**。
2. 两个 run 在时间上相隔很远（pid 相差 1383，中间跑了 14 个 run），
   所以这 1.2% 里**包含了机器状态漂移**（T4 降频、共享宿主争用），是一个偏保守的估计。
3. **回头看第 16 节**：`max_num_seqs` 64→128 只差 1.38%，**刚好压在噪声底上**，不能算显著；
   128→256 差 1.95%，略超噪声底。
4. Part I 第 12 节 #1 抱怨"没有重复实验"——**Kaggle 这一轮部分补上了这个缺口**，
   Part I 里那些 10% 上下的结论（如 9.2 的 11.2%）现在有了一个可参照的尺度。

## 14. Workload

prompt 文件与本机**完全共用**（Qwen2.5 各尺寸同一套 tokenizer），
生成命令与固定 seed=42 见 `docs/runbook_three_models.md` 第 3.2 节。

| 实验 | prompt 文件 | 条数 | 单条 token | 总 prefill | `max_model_len` |
| --- | --- | --- | --- | --- | --- |
| E1 基线 | `common.py` 默认 prompt | 1 / 16 | 10–12 | 10 / 176 | 1024 |
| E2 并发 | `concurrency/prompts_512x64.txt` | 64 | 512（互不相同）| 32768 | 1024 |
| E3 上下文 | `context_length/prompts_{512,2048,4096,8192,16384}.txt` | 16 / 4 / 2 / 1 / 1 | 512 / 2048 / 4096 / 8192 / 16384 | **固定 8192**（前四组）| 1024 / 2560 / 4608 / 8704 / 16896 |
| E4 prefix | `prefix_caching/prompts_{shared,random}_1024.txt` | 16 | 1152 | 18432 | 1280 |
| E5 chunked | `chunked_prefill/prompts_mixed.txt` | 16 | 3584 ×8 / 64 ×8 | 29184 | 4096 |
| E6 量化 | `common.py` 默认 prompt | 16 | 10–12 | 176 | 1024 |

- Sampling：`temperature=0.0`（贪心）、`seed=42`、`max_tokens=64`，与 Part I 一致。
- **prompt 文件按 `make_prompts.py --check` 在 Kaggle 上复核过长度**（runbook 第 5.1 节 Cell 6）。
  产物侧的核对：E3 各臂的 `input_tokens` 实测为 `512/2048/4096/8192/16384`，
  E2/E7 为 `512`，E4 为 `1152`，E5 为 `1824(3584)`——**与配置逐条吻合，不是配置上限**。
- Warmup：**离线 run 依然没有排除 warmup**，与 Part I 同样的局限（Part I 第 12 节 #3）。
- **E3 的 `max_model_len` 在 V0 下没有触发 Part I 坑 6**：坑 6 只在**显式传**
  `--max-num-batched-tokens` 且它小于 `max_model_len` 时才会崩；
  不传时 V0 自己会把单步预算设得足够大（E3 的 16384 臂 `max_model_len=16896`、未传预算，正常跑完）。
  这条是对 Part I 坑 6 的补充说明。

## 15. E1 HF vs vLLM

同 Part I，拆成两步把引擎差异与批处理差异分开。

### 15.1 E1a 单请求对单请求

| run | backend | batch 延迟 | output tok/s |
| --- | --- | --- | --- |
| `kaggle-1.5b-hf-single` | HF `model.generate`（逐条串行）| 2840.63 ms | 22.53 |
| `kaggle-1.5b-vllm-single` | vLLM，`max_num_seqs=1` | **1160.40 ms** | **55.15** |

**vLLM 延迟是 HF 的 0.408 倍（快 2.45×），吞吐 2.45 倍。**

**与本机对比时必须注意**：Part I 同一步在 0.5B 上是 **3.7×**（Part I 第 5.1 节）。
从 3.7× 降到 2.45× 说明什么？**不能归因**——涉及四个同时变化的变量（模型 3 倍大、
GPU 不同、dtype 不同、引擎 V1 vs V0）。合理的读法只有一句：
**"在 1.5B + T4 这个组合上，vLLM 相对 HF 的单请求优势仍然是 2 倍以上。"**

HF 的 `tpot_ms=44.38`（≈ 2840.63/64，按全部输出 token 摊），vLLM 单请求约 18.1 ms/token。

> **口径**：两边都是 batch=1，所以这一步是干净的引擎对比。
> 但 `hf-single` 的 `latency_mode=single` 是"HF 逐请求"语义，
> `vllm-single` 的 `single` 是"batch 内只有 1 条"语义——两者在 n=1 时重合。

### 15.2 E1b 批处理收益（vLLM 自己 1 → 16）

| run | requests | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- |
| `kaggle-1.5b-vllm-single` | 1 | 1160.40 ms | 55.15 tok/s |
| `kaggle-1.5b-vllm-batch16` | 16 | 1212.59 ms | **844.47 tok/s** |

**吞吐 ×15.3，而 batch 延迟只涨 4.5%（×1.045）。**

Part I 同一步是 **×16.6、延迟 ×0.965**（Part I 第 5.2 节）。
两台机器给出同一条结论：**1→16 这个区间里，批处理几乎是把闲置算力直接换成吞吐**。
T4 上延迟开始有 4.5% 的可观测上升（本机是 −3.5%），说明 16 条已经让 T4 略微吃紧
——这与第 16 节"T4 的饱和点在 64 附近"是一致的。

## 16. E2 并发扫描：饱和点就在 `max_num_seqs` = 请求数

64 个请求（512 token，互不相同），只改 `max_num_seqs`。

| `max_num_seqs` | 调度轮次 | batch 延迟 | batch 吞吐 | 每请求均摊 | 相对上一档 | 相对 64 |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 4 | 9263.90 ms | 442.15 tok/s | 144.75 ms | — | — |
| **64** | **1** | **6519.28 ms** | **628.29 tok/s** | **101.86 ms** | **×1.421** | — |
| 128 | 1 | 6610.40 ms | 619.63 tok/s | 103.29 ms | ×0.986 | −1.38% |
| 256 | 1 | 6741.62 ms | 607.57 tok/s | 105.34 ms | ×0.981 | −3.30% |

### 16.1 结论一：吞吐的拐点就是"一轮装下全部请求"的那一刻

`max_num_seqs=16` 时 64 个请求要分 **4 个调度轮次**；`max_num_seqs≥64` 时**一轮装完**。
吞吐的全部收益（×1.42）都发生在 16→64 这一步，之后**再无收益**。

**一旦 `max_num_seqs ≥ 请求数`，这个旋钮就失效了**——再往上加只是让引擎多留调度余量，
换不到任何吞吐。第 21 节会看到，`concurrency-64` 这个点其实已经**恰好落在饱和点上**。

### 16.2 结论二：64 之后是"平"还是"降"？——用噪声底判

13.5 测出的噪声底是 **±1.2%**。据此：

| 对比 | 吞吐变化 | 判定 |
| --- | --- | --- |
| 64 → 128 | −1.38% | **压在噪声底上，不能算显著** |
| 128 → 256 | −1.95% | 略超噪声底，但方向与预期相反 |
| 64 → 256 | −3.30% | 超出噪声底，**是可观测的轻微劣化** |

**没有出现 OOM**，`max_num_seqs=256` 配 1024 上下文、`gpu_memory_utilization=0.85`
在 T4 16GB 上跑得下。但**没有任何显存遥测**，所以"离上限还有多远"**无法回答**。

### 16.3 与本机的对比：本机那个"边际收益递减"其实是在逼近同一个天花板

Part I 的 E2 只扫到 64，观察到"16→32 是 ×1.351、32→64 只有 ×1.275"，
当时把它读成"接近饱和"（Part I 第 6 节）。**Kaggle 这一轮把这个猜测变成了观测**：

- 本机 64 个请求配 `max_num_seqs=64` → 也是"一轮装完"，**同样在饱和点上**；
- Kaggle 从 64 再往上扫，证明**天花板就在这里**，加宽只会轻微变差。

所以 Part I 第 11.2 节"把 `max_num_seqs` 开大，直到显存或边际收益不允许"这条建议，
现在可以写得更准确：**开到"等于并发请求数"即可，再大没有意义**。
本机 8GB 的 OOM 边界**仍然是未知**（Part I 第 12 节 #9 未解决）——
Kaggle 证明的是"没有收益"，不是"会 OOM"。

## 17. E3 上下文长度

总 prefill 固定 8192 token（前四行），只改单条长度与 batch 宽度。

| 单条长度 | batch 宽度 | 总 prefill | batch 延迟 | batch 吞吐 | 语义 |
| --- | --- | --- | --- | --- | --- |
| 512 | 16 | 8192 | 2364.44 ms | 433.08 tok/s | batch |
| 2048 | 4 | 8192 | 2346.75 ms | 109.09 tok/s | batch |
| 4096 | 2 | 8192 | 2373.16 ms | 53.94 tok/s | batch |
| 8192 | 1 | 8192 | 2748.77 ms | 23.28 tok/s | **single** |
| **16384** | 1 | **16384** | **5760.25 ms** | **11.11 tok/s** | **single** |

### 17.1 结论一：Part I 的核心发现被独立复现

前四行（总 prefill 都是 8192）延迟落在 **2346.75 – 2748.77 ms**。
只看 batch 宽度 ≥2 的三行：**2346.75 – 2373.16 ms，极差仅 1.13%**。

Part I 在同一设计下的极差是 **1.125%**（Part I 第 7.1 节）。

> **两台机器、两个模型、两个引擎、两种 dtype，得到同一个 1.1% 量级的极差。**
> 这是本报告里**最可复现的一条结论**：**总 prefill 量不变时，延迟与单条序列长度基本无关**，
> prefill 代价正比于**总 token 数**而非单条长度。

### 17.2 结论二：吞吐的暴跌依旧是 batch 宽度效应

吞吐 433.08 → 23.28（×0.054）看起来像"长上下文很慢"，但前四行总 prefill 相同、
延迟几乎相同（17.1），变的只有 batch 宽度 16→1 与输出总量 1024→64。
**这张表不能读成"上下文越长吞吐越低"**，与 Part I 第 7.2 节同一结论。

### 17.3 结论三：16384 的代价在 T4 上是线性的，本机是次线性的

8192 → 16384（batch 宽度同为 1，工作量翻倍）：

| | 8192 | 16384 | 倍数 |
| --- | --- | --- | --- |
| Part I（4060 / 0.5B / bf16 / FlashAttention）| 1475.21 ms | 2212.62 ms | **×1.50** |
| Part II（T4 / 1.5B / fp16 / XFormers）| 2748.77 ms | 5760.25 ms | **×2.10** |

**本机是次线性（2 倍 token 只花 1.5 倍时间），T4 是几乎正好线性（×2.10）。**

**候选解释（无法分离，均为假设）：**

1. **attention backend**：本机走 FlashAttention，T4 走 XFormers（13.2 问题 5）。
   长上下文里 attention 的占比随长度上升，一个较弱的 attention 实现会在这里被放大。
2. **模型规模**：1.5B 的线性层工作量是 0.5B 的 3 倍。但这一条解释不了
   "为什么倍数会随长度变化"——纯线性层的话，8192 和 16384 的比值应该与长度无关。
3. **dtype**：bf16 vs fp16，理论上不该有这个方向差异。

**三条里第 1 条最贴合"倍数随长度增长"这个现象，但本次没有 attention backend 的记录
（13.4 缺陷 3），也没有能分离模型规模与硬件的对照臂，所以只能说"候选"。**

### 17.4 一致性交叉检查（两个独立 run）

`context-512`（16 请求 × 512 token，`max_num_seqs=16`）是 2364.44 ms；
`concurrency-16`（64 请求 × 512 token，`max_num_seqs=16`）是 9263.90 ms。

64 请求在 `max_num_seqs=16` 下正好是 **4 个调度轮次**：

```text
4 × 2364.44 = 9457.76 ms   （若严格线性）
实测          9263.90 ms   （好 2.05%）
```

**两个不同 prompt 文件、不同 `requests`、不同实验批次的 run，在同一个
"512 token × 16 并发"的批次形态上吻合到 2%。** 这是 Part II 内部的一条独立自洽性证据，
说明离线 runner 的 batch 计时口径是稳定的。

## 18. E4 Prefix caching：一个未解释的异常

### 18.1 数据

四臂，16 条 × 1152 token（1024 前缀 + 各自不同的尾部 128），`max_num_seqs=16`、`max_model_len=1280`。

| workload | caching | batch 延迟 | batch 吞吐 | 相对 off |
| --- | --- | --- | --- | --- |
| **shared**（1024 逐字节相同的前缀）| off | 4064.95 ms | 251.91 tok/s | — |
| | **on** | **10134.27 ms** | **101.04 tok/s** | **延迟 ×2.493，吞吐 −59.9%** |
| **random**（1024 各自唯一的前缀）| off | 4106.13 ms | 249.38 tok/s | — |
| | on | 4133.09 ms | 247.76 tok/s | 延迟 ×1.007，吞吐 **−0.65%** |

workload 本身没有问题：prompt 文件实测 **shared 组前 5551/6207 字符逐字节相同，
random 组 0 字符相同**，两个 off 臂也吻合到 1.0%。

### 18.2 为什么这是异常

**它与 Part I 的同名实验方向相反**：

| | shared-on 相对 shared-off | random-on 相对 random-off |
| --- | --- | --- |
| Part I（4060 / 0.5B / V1）| **+21.9% 吞吐** | −1.2% |
| Part II（T4 / 1.5B / V0）| **−59.9% 吞吐** | −0.65% |

**两个结论不可能同时为真。** 而且 Kaggle 这边的实验设计**更干净**
（13.3(d)：V0 下 prefix caching 默认就是关的，off 臂不传参即为真 off）。

### 18.3 已排除的解释：机器状态漂移

run 的 pid 严格单调，可用来还原执行顺序：

| 顺序 | run | pid | batch 延迟 |
| --- | --- | --- | --- |
| 1 | `prefix-shared-off` | 1774 | 4064.95 ms |
| 2 | **`prefix-shared-on`** | **1862** | **10134.27 ms** |
| 3 | `prefix-random-off` | 1966 | 4106.13 ms |
| 4 | `prefix-random-on` | 2054 | 4133.09 ms |

**那个 2.49× 的慢 run 被两个 ~4.1 s 的快 run 前后夹住。**
如果是降频、宿主争用或"跑到一半机器变慢"，不可能只慢中间这一个 run。
**所以这是内容/配置相关的真实行为，不是环境噪声。**

### 18.4 候选原因（均为假设，未能区分）

1. **V0 的 prefix caching 命中路径本身慢**：缓存真的命中时，V0 需要引用/管理共享 block，
   可能退回到一条比 XFormers 常规路径更慢的 kernel。
   *支持*：只有 shared-on 命中缓存，而恰好只有它变慢——**这个方向是对的**。
   *存疑*：2.5× 的幅度太大，且 Part I 在 V1 上完全没看到这个现象。
2. **attention backend 回退**：若 V0 的 prefix caching 需要某个 backend 支持，
   开启时可能换掉 XFormers。*但*：那 random-on 也该受影响，实测没有。
3. **block manager 竞争**：16 条序列共享 1024 token（block_size=16 时约 64 个 block），
   引用计数与分配开销集中。*但*：这个开销量级上解释不了 6 秒。

**第 1 条最贴合数据，但都不足以定论。**

### 18.5 本节结论：空

> **在重跑并抓到 vLLM 启动日志（`enable_prefix_caching` 实际取值、所选 attention backend）
> 之前，Kaggle 的 prefix caching 结论记为"空"。**
> 不写"T4 上 prefix caching 是负收益"——那是把一个未解释的异常当成结论；
> 也不写"prefix caching 有效"——实测数字明确反对。

**重跑方案**（列入待办）：同一 workload 跑 shared-on 三次（测重复性），
同时保存 server 端的 vLLM 启动日志与 `--no-enable-prefix-caching` 对照。
若 server 模式可用（见第 22 节的绕过方案），per-request TTFT 能直接判定
"是缓存命中变慢"还是"整批被拖慢"。

## 19. E5 Chunked prefill：负收益，且比本机大得多

`prompts_mixed.txt`（8 条 3584 + 8 条 64，交替），`max_model_len=4096`、16 请求。
两臂都在 **V0** 上（T4 没有 V1，所以**没有 `v1ref` 臂**）。

| 臂 | `max_num_batched_tokens` | 切块 | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- | --- |
| `off` | 4096 | 否 | **6186.46 ms** | **165.52 tok/s** |
| `on` | 2048 | 是 | 11966.18 ms | 85.57 tok/s |

### 19.1 结论一：离线 batch 下开切块更慢，方向与本机一致

| | 延迟倍数 | 吞吐变化 |
| --- | --- | --- |
| Part I（V0，本机）| ×1.133 | −11.7% |
| Part II（V0，T4）| **×1.934** | **−48.3%** |

**方向相同（都是负收益），但 T4 上的代价差不多是本机的 4 倍。**

机制解释与 Part I 第 9.1 节相同，这里同样适用：
**离线模式 16 个请求同时入队，本来就不存在"短请求被长 prefill 堵住"的场景**，
切块只多了调度开销和中间状态管理。**切块是尾延迟优化，不是吞吐优化。**

**为什么 T4 上代价更大？** 候选：
T4 单卡算力弱、V0 的切块在 SM 7.5 + XFormers 上开销占比更高。
**这是假设**——本次没有 V1 对照臂、没有 backend 记录、也没有逐 step 的调度日志，无法证实。

### 19.2 未回答的问题（与本机同样未回答）

**"切块是否改善短请求的尾延迟"——T4 上同样没有测到**，因为 HTTP server 实验失败（第 22 节），
没有 per-request TTFT。所以：

> **Kaggle 的 E5 同样只能给出"离线吞吐"的结论，不能声称 chunked prefill 无效。**
> Part I 第 9.3 节和本节合起来说明：这个特性的卖点**在本项目里至今没有被测到过**。

### 19.3 配置口径（必须随结论引用）

- 两臂的 `max_num_batched_tokens` 不同（4096 vs 2048）是**必然的**，不是污染：
  V0 关掉切块时单条 prompt 必须在一个 step 里放下，预算必须 ≥ 最长 prompt 3584。
  这个差异本身就是"切块 vs 不切块"的操作定义（与 Part I 第 9 节一致）。
- 两臂 `max_model_len` 统一为 **4096**，最长 prompt 只有 3584，**实际用到的长度没变**。
- 产物侧核对：两臂 `input_tokens` 均为 `1824(3584)`，**完全一致**，可比。

## 20. E6 量化：本项目第一个可归因的量化结论

### 20.1 结果

| 臂 | checkpoint | batch 延迟 | batch 吞吐 | 相对 FP16 基线 |
| --- | --- | --- | --- | --- |
| FP16 基线 | `Qwen/Qwen2.5-1.5B-Instruct` | 1212.59 ms | 844.47 tok/s | — |
| **AWQ** | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | **1070.26 ms** | **956.78 tok/s** | **延迟 ×0.883，吞吐 +13.3%** |

### 20.2 为什么这一节比 Part I 可信

**两个臂的配置逐项对齐**（全部来自产物，不是手工声明）：

| 维度 | FP16 基线 | AWQ | 对齐？ |
| --- | --- | --- | --- |
| 引擎 | V0 | V0 | ✅（Kaggle 全部 V0，13.3）|
| `dtype` | half | half | ✅ |
| `gpu_memory_utilization` | 0.85 | 0.85 | ✅ |
| prompt | `common.py` 默认（10–12 token）| 同左 | ✅ |
| `requests` / `max_num_seqs` | 16 / 16 | 16 / 16 | ✅ |
| `max_model_len` / `max_tokens` | 1024 / 64 | 1024 / 64 | ✅ |
| `seed` | 42 | 42 | ✅ |
| `tensor_parallel_size` | 1 | 1 | ✅ |
| **唯一变量** | FP16 权重 | **4-bit AWQ 权重** | — |

实测 `input_tokens` 两臂都是 `11(12)`，**逐条一致**。

> 对比 Part I 第 10.2 节的三个混淆变量（引擎、显存配额、prompt 全不一致，13% 归因不了），
> **Kaggle 这一节把那个坑填上了**：13.3% 的收益可以干净地归因给量化本身。
> 讽刺的是，帮上忙的正是那个"限制"——T4 不支持 V1，反而让所有臂被迫同引擎。

### 20.3 结论与边界

1. **AWQ 在 T4 + 1.5B 上带来 +13.3% 吞吐、延迟降到 0.883 倍。**
2. **这是收益的下界，不是上界。** T4 是 SM 7.5，**没有 Marlin kernel**（需要 SM 8.0+），
   AWQ 只能走较慢的解量化路径。在支持 Marlin 的硬件上，收益应当更大。
3. **机制上没有惊喜**：batch=16、prompt 只有 10 token，decode 阶段是**显存带宽受限**的，
   权重从 16 bit 压到 4 bit 直接减少了每步要读的字节数。
4. **本节的收益是"速度"，不是"显存"。**

### 20.4 显存：仍然无数据

**Kaggle 侧一个 GPU 遥测文件都没有**，也没有 KV cache usage 指标。按项目规则
（不用总显存占用替代 KV cache usage），本节**不提供任何实测显存数字**。

可以做一次**算术推算**，但必须标明它不是测量值：

```text
FP16 权重：1.5B × 2 B            ≈ 3.1 GB
AWQ  权重：1.5B × 0.5 B + 开销   ≈ 0.8 – 1.3 GB
推算节省                          ≈ 1.8 – 2.3 GB
```

**这是按位宽推算的，不是实测。** 要回答"量化省了多少显存"，必须补一次带
`scripts/collect_gpu_metrics.py` 的对照 run（列入待办）。

### 20.5 未执行 / 无数据

| 项 | 状态 |
| --- | --- |
| GPTQ on T4 | **未执行** |
| fp8 KV cache | **T4 不支持**（需 SM 8.9+），本机才有这一臂 |
| 量化臂的显存对比 | **无数据**（未采 GPU 遥测）|
| AWQ 输出 sanity check | **未做**——`--include-text` 已传，产物里有 `text` 字段，但本次未逐条比对 |

## 21. E7 TP=1 vs TP=2：+24.8%，但每卡效率掉了 38%

同 E2 的 64 请求 × 512 token workload，`max_num_seqs=64`，只改 `--tensor-parallel-size`。

| 臂 | GPU 数 | batch 延迟 | batch 吞吐 | 单卡均摊吞吐 |
| --- | --- | --- | --- | --- |
| `tp1` | 1 | 6596.69 ms | 620.92 tok/s | **620.92** |
| `tp2` | 2 | **5286.51 ms** | **774.80 tok/s** | **387.40** |
| 变化 | ×2 | ×0.801 | **+24.8%** | **−37.6%** |

### 21.1 结论一：TP=2 是有效加速，而且推翻了 runbook 的预设

`docs/runbook_three_models.md` 第 5.2 节在实验**之前**写下的预期是：

> "T4 之间走 PCIe、没有 NVLink，TP=2 的通信开销可能吃掉收益。
> **这正是要测的东西**：小模型 + 慢互联，TP=2 很可能比 TP=1 更慢。"

**实测结果是反的：TP=2 快 24.8%。** 这个预设值得保留在文档里——它是一条被数据推翻的假设。

**为什么预设错了**（物理解释）：1.5B 太小，decode 阶段**瓶颈是显存带宽而非算力**。
TP=2 把权重切到两张卡上，**每张卡每步只需读一半的权重**，带宽瓶颈直接减半；
而每层要做的 all-reduce（hidden 1536、28 层）在 PCIe 上虽然慢，但**总量很小**，
被权重读取的节省盖过了。**"慢互联"的直觉适用于算力受限的大模型，不适用于带宽受限的小模型。**

### 21.2 结论二：但每卡效率下降 37.6%，吞吐导向的部署不该选 TP

774.80 tok/s 用了 **2 张卡**。按单卡均摊：620.92 → 387.40，**掉了 37.6%**。

**工程读数**：如果目标是在 2×T4 上最大化总吞吐，**跑两个独立单卡实例**
（runbook 的 E7b）应当优于 TP=2——每个实例各自 620.92 tok/s，合计约 1241.86 tok/s，
是 TP=2 的 1.60 倍。**代价是每个实例要各自装一份权重、各自吃一份 KV cache 配额。**

> **E7b（两个独立单卡实例）未执行**，所以上面这 1241.86 是**推算，不是实测**
> ——它假设两个实例互不干扰，而实际会共享 PCIe 带宽与宿主 CPU。
> **这正是 E7b 值得跑的理由。**
>
> 项目规则（`README.md` 常见限制）：**TP=2、两个单卡实例、单卡运行是三种不同实验**，
> 结果不能混排。本节只测了第一种和第三种。

### 21.3 口径与交叉验证

- `tp1` 就是 13.5 节那个重复实验的一半，它与 `concurrency-64` 相差 1.19%，**在噪声底内**，
  所以 `tp2` 的 +24.8% **远高于噪声**，可以放心解释。
- `CUDA_VISIBLE_DEVICES=0,1` 的实际取值**没有被记录**（本次之前 metadata 不记环境变量）。
  `tensor_parallel_size=2` 在产物里是明确记录的，所以"用了两张卡"这一点可自证。

## 22. E8 HTTP server：失败实验（保留）

`results/raw/kaggle-1.5b-http-concurrency-16.jsonl`，**64/64 请求全部失败**：

```text
status: failed      ok: 0      failed: 64
errors: APIConnectionError: Connection error.   ×64
wall_ms: 5751.35    output_tokens: 0
mean/p50/p95/p99 TTFT、TPOT、latency: 全部 null
```

### 22.1 原因

**Kaggle Notebook 不允许后台进程。** runbook 第 5.3 节用的是：

```bash
!CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server ... &
```

`nohup ... &` 起的 server 随 cell 结束被回收，压测时 `127.0.0.1:8000` 上没有任何进程监听，
于是每个请求都在**连接阶段**就失败（`wall_ms=5751` 是 64 个连接错误的总耗时，不是推理耗时）。

### 22.2 后果：Kaggle 侧完全没有 per-request 指标

| 缺失 | 影响 |
| --- | --- |
| TTFT / TPOT / p95 / p99 | E4 的缓存收益、E5 的尾延迟**在 T4 上同样无法评估**（第 18、19 节都因此结论受限）|
| `req/s` | 无法与离线 batch 吞吐做交叉验证——**Part I 第 5.3 节那种"两条独立路径互验"在 Kaggle 上做不了** |
| GPU 遥测 | 第 16、20 节的显存问题全部无解 |

> **Part I 与 Part II 合起来看，本项目至今没有任何一组 T4 上的 per-request 数字。**
> 这是 Part II 最大的缺口，优先级高于补 7B。

### 22.3 可尝试的绕过方案（未验证）

Kaggle 不允许**跨 cell 常驻**的后台进程，但可以把 server 的整个生命周期**关在一个 cell 内**：
用 `subprocess.Popen` 启动 server → 轮询 `/v1/models` 直到就绪 → 跑 `benchmark_http.py` → `terminate()`。

```python
import subprocess, time, urllib.request
proc = subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-1.5B-Instruct", "--served-model-name", "qwen1.5b",
    "--host", "127.0.0.1", "--port", "8000",
    "--max-model-len", "1024", "--max-num-seqs", "16",
    "--gpu-memory-utilization", "0.85", "--no-enable-prefix-caching",
], env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "VLLM_USE_V1": "0"})
for _ in range(120):                       # 最多等 6 分钟
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)
        break
    except Exception:
        time.sleep(3)
# 然后 !python scripts/benchmark_http.py ... ，最后 proc.terminate()
```

**这条路径的关键是"不要跨 cell"**：server 与压测必须在同一个 cell 的执行期内完成。
**本次未验证**，列入待办。

## 23. 跨硬件可比性 & Part II 局限

### 23.1 什么可以比，什么不可以

**不可以比（Part I ↔ Part II 的任何绝对数字）：**

四个混淆变量同时存在，**没有任何一对 run 是只差一个变量的**：

| 变量 | Part I（本机 0.5B）| Part II（Kaggle 1.5B）|
| --- | --- | --- |
| GPU | RTX 4060 Laptop 8GB（Ada, SM 8.9）| Tesla T4 16GB（Turing, **SM 7.5**）|
| 模型 | Qwen2.5-**0.5B** | Qwen2.5-**1.5B**（3 倍参数）|
| dtype | **bfloat16** | **float16**（T4 无 BF16）|
| 引擎 | **V1**（默认）| **V0**（T4 不支持 V1）|
| attention backend | FlashAttention | XFormers |

**可以比（定性、方向性的结论）：**

- **E3 的"总 prefill 固定 → 延迟与单条长度无关"**：两台机器极差都是 ~1.1%
  （Part I 1.125%、Part II 1.13%），**这是本项目最可复现的一条**（17.1）。
- **E1b 的"1→16 批处理收益接近线性"**：×16.6 vs ×15.3（15.2）。
- **E2 与 Part I 的 7.2/6 节："吞吐随 batch 宽度上升"**，且 Kaggle 补齐了本机的天花板（16.3）。
- **E4/E5 的"离线 batch 测不到 chunked prefill 的卖点"**：两台机器同结论（19.2）。

### 23.2 Part II 的局限（与 Part I 第 12 节并列）

| # | 局限 | 影响 |
| --- | --- | --- |
| 1 | **引擎版本只来自操作者陈述**（13.3）| "全部 V0"无法从 22 个产物自证。补丁已打，对已有文件无效 |
| 2 | **Transformers 版本不可考，环境快照与实际矛盾**（13.4）| `env-kaggle-1.5b.json` 不可信 |
| 3 | **attention backend 无记录**（13.4）| 17.3 的"16384 代价"无法归因 |
| 4 | **Kaggle 侧零 GPU 遥测** | E2/E6 的显存问题全部无解（16.2、20.4）|
| 5 | **Kaggle 侧零 per-request 指标**（第 22 节）| E4/E5 的尾延迟问题在 T4 上同样未答；无 server/离线互验 |
| 6 | **E4 的 prefix caching 异常未解释**（第 18 节）| 该节结论为空；且它与 Part I 的 +21.9% **直接矛盾**，必有一方的解释是错的 |
| 7 | **7B 与 E7b（双实例）未执行** | 三模型矩阵完成 2/3；21.2 的推算未验证 |
| 8 | **每个配置只跑 1 次**，唯一例外是 13.5 的重复对 | 噪声底 ±1.2% 已测出，但大多数臂仍无重复 |
| 9 | **量化缺 GPTQ、缺显存、缺输出 sanity check**（20.5）| 量化一节只有速度结论 |
| 10 | **prompt 为程序生成** | 与 Part I 同一局限，绝对吞吐不能外推到生产流量 |

### 23.3 Part II 的三条主要收获

尽管局限很多，Kaggle 这一轮给出了三条 Part I 拿不到的结论：

1. **E6 量化首次可归因**（第 20 节）：+13.3% 吞吐、同引擎同配额同 prompt，
   把 Part I 第 10.2 节"归因不了"的坑填上了。**代价是 T4 没有 Marlin，这个数是下界。**
2. **E2 找到了 `max_num_seqs` 的天花板**（第 16 节）：**等于并发请求数**，
   把 Part I"边际收益递减"的猜测变成了观测。7B 与其它 batch 宽度的实验可以直接照用。
3. **E7 推翻了预设**（第 21 节）：TP=2 在小模型 + 慢互联上是**加速**而非减速，
   但同时暴露了每卡效率掉 37.6%——**"总吞吐"和"每卡吞吐"是两个不同的问题**。

以及一条方法论收获：**13.5 的噪声底 ±1.2%**，让本报告里所有小于这个幅度的差异
（如 E2 的 64→128）都能被正确地判为"不可解释"。

