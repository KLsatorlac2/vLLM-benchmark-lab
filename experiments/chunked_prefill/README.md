# Chunked Prefill

使用长输入/短输出和混合请求，比较 `--enable-chunked-prefill` 开关。重点观察短请求 TTFT 和 p95 是否受到长 prefill 影响。

## 本轮结果（2026-09-26，Qwen2.5-0.5B，RTX 4060）

`prompts_mixed.txt`：8 条 3584 token + 8 条 64 token 交替，共 16 请求。
`max_model_len=4096`、`max_tokens=64`、`gpu_memory_utilization=0.75`。

| 臂 | 引擎 | `max_num_batched_tokens` | 切块 | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- | --- | --- |
| `off` | V0 | 4096 | 否 | **2196.55 ms** | **466.19 tok/s** |
| `on` | V0 | 2048 | 是 | 2487.99 ms | 411.58 tok/s |
| `v1ref` | V1 | 2048 | 恒开 | 2237.98 ms | 457.56 tok/s |

### 结论

1. **离线 batch 下开切块反而更慢（−11.7%）。** 离线模式所有 16 个请求同时入队，
   本来就不存在"短请求被长 prefill 堵住"的场景；切块却多了调度开销和中间状态管理，
   换不到插队收益。**切块是尾延迟优化，不是吞吐优化。**
2. **V1 的切块比 V0 快 11.2%**（`v1ref` vs `on`，两臂切块配置相同、只差引擎版本），
   且与 `off` 几乎持平（−1.9%）。V1 把 chunked prefill 做成了默认路径，实现更好。
3. **"切块是否改善短请求尾延迟"——本轮没有测到。** 需要 server 模式的 per-request TTFT
   （见 `docs/runbook_three_models.md` 4.7 节）。**所以不能声称 chunked prefill 无效，
   只能说本实验设计观察不到它的卖点。**

### 配置口径（必须随结论引用）

- 两臂的 `max_num_batched_tokens` 不同（4096 vs 2048）是**必然的**，不是污染：
  V0 关掉切块时单条 prompt 必须在一个 step 里放下，预算必须 ≥ 最长 prompt 3584。
- 三个臂的 `max_model_len` 统一取 **4096**。原设计的 4608 会在 V0 下**启动即崩**
  （`max_num_batched_tokens (4096) is smaller than max_model_len (4608)`）——
  见 `docs/runbook_three_models.md` 坑 6。最长 prompt 只有 3584，**实际用到的长度没有变**。
- **`off`/`on` 跑在 V0，`v1ref` 跑在 V1**，跨臂比较时必须标注引擎版本。

分析见 `docs/benchmark_report.md` 第 9 节。

## Kaggle T4 ×2 / Qwen2.5-1.5B（2026-09-27）

同一份 `prompts_mixed.txt`（8 条 3584 + 8 条 64 交替），`max_model_len=4096`、
`max_tokens=64`、`dtype=half`。两臂都在 **V0** 上
（**T4 没有 V1，所以没有 `v1ref` 臂**）。

| 臂 | `max_num_batched_tokens` | 切块 | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- | --- |
| `off` | 4096 | 否 | **6186.46 ms** | **165.52 tok/s** |
| `on` | 2048 | 是 | 11966.18 ms | 85.57 tok/s |

### 结论一：负收益，方向与本机一致但代价大得多

| | 延迟倍数 | 吞吐变化 |
| --- | --- | --- |
| 本机（V0）| ×1.133 | −11.7% |
| T4（V0）| **×1.934** | **−48.3%** |

**方向相同，但 T4 上的代价差不多是本机的 4 倍。**

机制解释与本机相同：**离线模式 16 个请求同时入队，本来就不存在"短请求被长 prefill 堵住"
的场景**，切块只多了调度开销和中间状态管理。**切块是尾延迟优化，不是吞吐优化。**

**为什么 T4 上代价更大？** 候选：T4 算力弱、V0 的切块在 SM 7.5 + XFormers 上开销占比更高。
**这是假设**——没有 V1 对照臂、没有 backend 记录、也没有逐 step 调度日志，无法证实。

### 结论二：本机那条 V1 结论在 T4 上无法验证

本机有一条"**V1 的切块比 V0 快 11.2%**"（靠 `v1ref` 臂得出）。
**T4 上 V1 不可用，所以这条无法交叉验证。**

### 未回答的问题（与本机同样未回答）

**"切块是否改善短请求尾延迟"——T4 上同样没测到。**
Kaggle 的 HTTP server 实验失败（不允许后台进程），没有 per-request TTFT。

> 本机第 9 节 + Kaggle 这一节合起来说明：
> **这个特性的卖点在本项目里至今没有被测到过。**
> 不能声称 chunked prefill 无效，只能说实验设计观察不到它的卖点。

### 配置口径（必须随结论引用）

- 两臂的 `max_num_batched_tokens` 不同（4096 vs 2048）是**必然的**，不是污染：
  V0 关掉切块时单条 prompt 必须在一个 step 里放下，预算必须 ≥ 最长 prompt 3584。
- 两臂 `max_model_len` 统一 4096，最长 prompt 只有 3584，**实际用到的长度没变**。
- 产物侧核对：两臂 `input_tokens` 均为 `1824(3584)`，**完全一致**，可比。

分析见 `docs/benchmark_report.md` Part II 第 19 节。