# Quantization

先建立 FP16/BF16 基线，再验证 AWQ、GPTQ 和 FP8 KV cache。记录 checkpoint、硬件、vLLM 版本、显存、速度和输出 sanity check；不支持的组合保留失败日志。

## 本轮结果（2026-09-26，Qwen2.5-0.5B，RTX 4060）

同一份默认 prompt（10–12 token）、16 请求、`max_tokens=64`、`max_model_len=1024`。

| 臂 | checkpoint / 设置 | 引擎 | `gpu_mem_util` | batch 延迟 | batch 吞吐 | 相对 FP16 |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 基线 | `Qwen2.5-0.5B-Instruct` | V1 | 0.90 | 952.11 ms | 1075.51 tok/s | — |
| AWQ | `Qwen2.5-0.5B-Instruct-AWQ` | 未记录（推定 V1）| 0.75 | 1097.54 ms | 932.99 tok/s | −13.3% |
| FP8 KV cache | `kv_cache_dtype=fp8` | **V0** | 0.75 | 1094.47 ms | 935.61 tok/s | −13.0% |

### 结论：本轮没有有效的量化结论

**这三条数字不能互比**，三个混淆变量都没对齐：

1. **引擎版本**：`kvfp8` 确定跑在 V0（V1 未实现 fp8 KV cache，抛 `NotImplementedError`，见坑 7），
   基线跑在 V1，`awq` 的引擎版本**无法从 JSONL 确认**——因为 **`VLLM_USE_V1` 没有被写进 metadata**。
   E5 已实测 V0/V1 之间本身就有约 10% 差距，和这里的 13% 同量级。
2. **`gpu_memory_utilization`**：基线 0.90，量化臂 0.75。
3. **prompt**：`runbook` 建议的"同参数基线" `concurrency-16` 用的是 512 token prompt，
   而量化臂用默认 prompt（10 token），**差 50 倍，batch 延迟根本不可比**。
   本表已改用同 prompt 的 `vllm-batch16`，但变量 1、2 仍在。

**正确写法**："量化臂在本机 0.5B 上没有可观测的吞吐收益"，
**不能写**"fp8/AWQ 比 FP16 慢 13%"。

### 输出 sanity check

两臂都用 `--include-text` 保存了文本，与同一份 prompt 逐条比对：

- **都是连贯英文**，正确回答了 4 个 prompt，没有乱码、没有复读崩溃。
- 两臂文本**互不相同（16/16 条）**，这是**预期**的：AWQ 改权重，fp8 改 KV 精度，
  两者都会改变数值轨迹，不是 bug。
- `kvfp8` 有一条出现明显重复句式（"I have a question about..." 连写数遍），是质量退化的信号，
  但**只有 1 条样本，不足以定性**，需要更大规模评测（困惑度 / 任务准确率）。

### 未执行 / 无数据

| 项 | 状态 |
| --- | --- |
| GPTQ | **未执行**（本机未验证 checkpoint）|
| 量化臂的显存占用 | **无数据**（未采 GPU 遥测，也没有 KV cache usage 指标）|
| AWQ / GPTQ / fp8 KV on T4 | **未执行**（Kaggle 部分未开始）|
| 与 FP16 对齐的基线（同引擎、同显存配额）| **未跑**，这是最该补的一条 |

分析见 `docs/benchmark_report.md` 第 10 节。