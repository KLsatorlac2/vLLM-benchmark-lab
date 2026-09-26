# Benchmark Report

本报告记录 2026-09-25 / 09-26 在本机 RTX 4060 上完成的 Qwen2.5-0.5B 全矩阵实验（E1–E6）。
所有数字都来自 `results/raw/local-0.5b-*.jsonl`。

> **数据文件不在版本库中**（`.gitignore` 排除了 `results/raw/*` 与 `results/processed/*`，
> 只保留 `.gitkeep` 目录占位）。文中每一处数字都写明了它来自哪个 raw 文件，
> 在有数据的机器上可用 `scripts/summarize_results.py` 重建聚合表并逐条核对。

**Kaggle 部分（1.5B / 7B / TP / 双卡）尚未开始**，第 2 节环境表和第 12 节限制里都按"未执行"标注。
本文中凡是只有本机数据的结论，都不能外推到 T4。

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
| Kaggle T4 ×2 | **未执行** |

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
workload 分布记在 `experiments/*/prompts_*.txt.manifest.json`。

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
需要 server 模式 + per-request TTFT（第 4.7 节）才能观测，本轮未执行。
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
   **Kaggle 部分本轮完全未执行**，所以本报告**没有任何跨硬件结论**。
2. **离线 batch 数字与 server per-request 数字不能混在同一张表里比**。
   本报告已分开（第 5–10 节是 batch，5.3 / 6.1 是 server）。
3. **TP=2、两个单卡实例、单卡运行是三种不同实验**。本机只有 1 张卡，**E7 跳过**。
4. **chunked prefill 的结论限于 V0**；V1 下该特性恒开，不能关闭。
5. **每条结论只对应一个 run，没有重复实验**，见第 12 节。

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
| 8 | **Kaggle 1.5B / 7B / TP 全部未执行** | 三模型矩阵只完成了 1/3；无跨硬件结论 |
| 9 | **E2 未扫到 OOM** | 8GB 的 `max_num_seqs` 上限未知 |
| 10 | **prompt 为程序生成** | 非真实语料，绝对吞吐不能外推到生产流量 |

优先级最高的两条是 **#2**（补记录，否则已有结果无法归因）和 **#4/#6**（补遥测和 server 测量，
否则量化和 chunked prefill 两节实际上没有有效结论）。待办清单见
`docs/runbook_three_models.md` 第 10 节。
