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

## Kaggle T4 ×2 / Qwen2.5-1.5B（2026-09-27）⚠️ **异常，结论为空**

同一设计：16 条 × 1152 token（1024 前缀 + 各自不同的尾部 128），
`max_num_seqs=16`、`max_model_len=1280`、`dtype=half`、**引擎 V0**。

| workload | caching | batch 延迟 | batch 吞吐 | 相对 off |
| --- | --- | --- | --- | --- |
| **shared**（1024 逐字节相同的前缀）| off | 4064.95 ms | 251.91 tok/s | — |
| | **on** | **10134.27 ms** | **101.04 tok/s** | **延迟 ×2.493，吞吐 −59.9%** |
| **random**（1024 各自唯一的前缀）| off | 4106.13 ms | 249.38 tok/s | — |
| | on | 4133.09 ms | 247.76 tok/s | 延迟 ×1.007，吞吐 **−0.65%** |

### ⚠️ 这是一个未解释的异常

**它与本机同实验方向相反：**

| | shared-on 相对 shared-off | random-on 相对 random-off |
| --- | --- | --- |
| 本机（4060 / 0.5B / V1）| **+21.9% 吞吐** | −1.2% |
| T4（1.5B / V0）| **−59.9% 吞吐** | −0.65% |

**两个结论不可能同时为真。**

### 已排除：机器状态漂移

run 的 pid 严格单调，可还原执行顺序：

| 顺序 | run | pid | batch 延迟 |
| --- | --- | --- | --- |
| 1 | `prefix-shared-off` | 1774 | 4064.95 ms |
| 2 | **`prefix-shared-on`** | **1862** | **10134.27 ms** |
| 3 | `prefix-random-off` | 1966 | 4106.13 ms |
| 4 | `prefix-random-on` | 2054 | 4133.09 ms |

**那个 2.49× 的慢 run 被两个 ~4.1 s 的快 run 前后夹住。**
降频、宿主争用这类环境噪声不可能只影响中间一个 run。
**所以这是内容/配置相关的真实行为。**

### workload 本身没有问题

prompt 文件实测：**shared 组前 5551/6207 字符逐字节相同，random 组 0 字符相同**；
两个 off 臂也吻合到 1.0%。设计是对的。

### 候选原因（均为假设，未能区分）

1. **V0 的 prefix caching 命中路径本身慢**——只有 shared-on 真的命中缓存，
   而恰好只有它变慢，**方向是对的**；但 2.5× 的幅度太大，且本机 V1 上完全没看到。
2. **attention backend 回退**——若开启 caching 会换掉 XFormers，
   那 random-on 也该受影响，**实测没有**。
3. **block manager 竞争**——16 条序列共享约 64 个 block 的引用计数开销，
   **量级上解释不了 6 秒**。

### 结论：空

> **在重跑并抓到 vLLM 启动日志之前，这一节记为"空"。**
> 不写"T4 上 prefix caching 是负收益"——那是把未解释的异常当成结论；
> 也不写"prefix caching 有效"——实测数字明确反对。

**重跑方案**：同 workload 跑 shared-on 三次（测重复性）+ 保存 server 端启动日志
（确认 `enable_prefix_caching` 实际取值与所选 attention backend）。
若 server 模式可用，per-request TTFT 能直接判定"是缓存命中变慢"还是"整批被拖慢"。

分析见 `docs/benchmark_report.md` Part II 第 18 节。