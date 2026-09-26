# Concurrency

固定模型和实际 prompt 长度，扫描 `max_num_seqs=16/64/256`。记录吞吐、p95 TTFT、p95 latency、显存和 OOM 点。HTTP server 压测结果需单独标记 `server_mode=true`。

## 本轮结果（2026-09-26，Qwen2.5-0.5B，RTX 4060）

64 个请求（`prompts_512x64.txt`，512 token 且互不相同），`max_model_len=1024`、
`max_tokens=64`、`gpu_memory_utilization=0.75`，只改 `max_num_seqs`：

| `max_num_seqs` | batch 延迟 | batch 吞吐 | 每请求均摊 | 相对上一档 |
| --- | --- | --- | --- | --- |
| 16 | 4073.41 ms | 1005.55 tok/s | 63.65 ms | — |
| 32 | 3015.13 ms | 1358.48 tok/s | 47.11 ms | ×1.351 |
| 64 | **2364.97 ms** | **1731.95 tok/s** | **36.95 ms** | ×1.275 |

- **吞吐 ×1.72（16→64），端到端延迟 ×0.58**，但**边际收益递减**（16→32 明显大于 32→64）。
- **`max_num_seqs=64` 没有 OOM**，8GB 跑得下。
- **128 / 256 未扫**，所以**显存上限点和 OOM 边界仍然是未知**——这是本实验最大的缺口。
- **本轮没有伴随 GPU 采集器**，各档显存占用**无数据**。

### Server 模式对照（`local-0.5b-http-concurrency-16`）

同一份 workload，并发 16，走 OpenAI 兼容 server（2 条 warmup 已排除）：

```text
req/s 14.71   output 941.23 tok/s   wall 4351.74 ms
TTFT  mean 206.92  p50 231.31  p95 368.93  p99 374.02 ms
TPOT  mean  13.79  p95  16.11 ms
lat    mean 1075.70 p50 1060.58 p95 1206.04 p99 1212.22 ms
```

离线 `concurrency-16` 是 1005.55 tok/s，server 是 941.23 tok/s（**比值 0.936**）。
两条独立代码路径在 6% 内吻合，互相验证。**只有这一组是 per-request 语义**，
其余全是 batch 级，不能混排。

GPU 遥测见 `results/raw/gpu-local-0.5b-http-16.jsonl`（峰值显存 4382/8188 MB = 53.5%，
最大利用率 85%，最大功耗 54 W——注意该 run 用的是 `gpu-memory-utilization=0.6`，
数字不能外推到 0.75/0.90 的配置）。

分析见 `docs/benchmark_report.md` 第 6 节。