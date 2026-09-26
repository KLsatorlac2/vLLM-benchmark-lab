# Prefix Caching

构造重复系统前缀和随机前缀两类 workload，对比 `--enable-prefix-caching` 开关。只有在前缀复用场景下解释缓存收益。

## 本轮结果（2026-09-26，Qwen2.5-0.5B，RTX 4060）

两个 workload 各跑开关两臂，共四组。16 条 × 1152 token（1024 前缀 + 各自不同的尾部 128），
`max_model_len=1280`、`max_num_seqs=16`、`max_tokens=64`、`gpu_memory_utilization=0.75`。

| workload | caching | batch 延迟 | batch 吞吐 | 相对 off |
| --- | --- | --- | --- | --- |
| **shared**（1024 逐字节相同的前缀）| off | 1613.21 ms | 634.76 tok/s | — |
| | **on** | **1323.59 ms** | **773.65 tok/s** | **延迟 ×0.820，吞吐 +21.9%** |
| **random**（1024 各自唯一的前缀）| off | 1579.45 ms | 648.33 tok/s | — |
| | on | 1597.91 ms | 640.84 tok/s | 延迟 ×1.012，吞吐 **−1.2%** |

### 结论

1. **前缀真共享时，缓存有明确收益：吞吐 +21.9%。**
2. **前缀不共享时，开关缓存毫无收益（−1.2%，噪声范围）。** `random` 臂是对照组，
   它证明了收益来自前缀复用，而不是"打开某个开关就变快"。
3. **两个 off 臂几乎相同**（1613.21 vs 1579.45 ms，差 2%），说明两个 workload 除前缀共享外是对等的，
   归因可以干净地落在这一点上。
4. **收益（+21.9%）远低于可缓存比例（1024/1152 = 88.9%）**：
   prefill 只是总工作量的一部分，16 条请求还要各自 decode 64 token，这部分不受影响。

### 口径

本节是**离线 batch 吞吐**，不是 per-request TTFT。
prefix caching 对**单请求 TTFT** 的影响需要 server 模式，本轮未做。

`--enable-prefix-caching` 必须显式传；V1 下默认是 **开**，省略参数会静默变成"开"
（见 `docs/runbook_three_models.md` 坑 2）。离线 runner 显式传了 False，所以 off 臂是可信的。

分析见 `docs/benchmark_report.md` 第 8 节。