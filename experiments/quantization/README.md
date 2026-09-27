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

## Kaggle T4 ×2 / Qwen2.5-1.5B（2026-09-27）✅ **本项目第一个可归因的量化结论**

同一份默认 prompt（10–12 token）、16 请求、`max_tokens=64`、`max_model_len=1024`、
`dtype=half`、**引擎 V0**、`gpu_memory_utilization=0.85`。

| 臂 | checkpoint | batch 延迟 | batch 吞吐 | 相对 FP16 基线 |
| --- | --- | --- | --- | --- |
| FP16 基线 | `Qwen/Qwen2.5-1.5B-Instruct` | 1212.59 ms | 844.47 tok/s | — |
| **AWQ** | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | **1070.26 ms** | **956.78 tok/s** | **延迟 ×0.883，吞吐 +13.3%** |

### 为什么这一节比本机可信：变量逐项对齐

| 维度 | FP16 基线 | AWQ | 对齐？ |
| --- | --- | --- | --- |
| 引擎 | **V0** | **V0** | ✅（Kaggle 全部 V0）|
| `dtype` | half | half | ✅ |
| `gpu_memory_utilization` | 0.85 | 0.85 | ✅ |
| prompt | 默认（10–12 token）| 同左 | ✅ |
| `requests` / `max_num_seqs` | 16 / 16 | 16 / 16 | ✅ |
| `max_model_len` / `max_tokens` | 1024 / 64 | 1024 / 64 | ✅ |
| `seed` / `tensor_parallel_size` | 42 / 1 | 42 / 1 | ✅ |
| **唯一变量** | FP16 权重 | **4-bit AWQ 权重** | — |

实测 `input_tokens` 两臂都是 `11(12)`，**逐条一致**。

> **对比本机第 10 节**：那里三个臂的引擎、显存配额、prompt **全不一致**，
> 13% 的差**归因不了**。Kaggle 这一节把那个坑填上了——
> **讽刺的是，帮上忙的正是那个"限制"：T4 不支持 V1，反而让所有臂被迫同引擎。**

### 结论与边界

1. **AWQ 在 T4 + 1.5B 上带来 +13.3% 吞吐、延迟降到 0.883 倍。**
2. **这是收益的下界，不是上界。** T4 是 SM 7.5，**没有 Marlin kernel**（需 SM 8.0+），
   AWQ 只能走较慢的解量化路径。支持 Marlin 的硬件上收益应当更大。
3. **机制上没有惊喜**：batch=16、prompt 只有 10 token，decode 阶段是**显存带宽受限**的，
   权重从 16 bit 压到 4 bit 直接减少了每步要读的字节数。
4. ⚠️ **本节的收益是"速度"，不是"显存"。**

### 显存：仍然无数据

**Kaggle 侧一个 GPU 遥测文件都没有**，也没有 KV cache usage 指标。
按项目规则（不用总显存占用替代 KV cache usage），**本节不提供任何实测显存数字**。

只做**算术推算**，且必须标明它不是测量值：

```text
FP16 权重：1.5B × 2 B            ≈ 3.1 GB
AWQ  权重：1.5B × 0.5 B + 开销   ≈ 0.8 – 1.3 GB
推算节省                          ≈ 1.8 – 2.3 GB
```

**要回答"量化省了多少显存"，必须补一次带 `collect_gpu_metrics.py` 的对照 run。**

### 未执行 / 无数据

| 项 | 状态 |
| --- | --- |
| GPTQ on T4 | **未执行** |
| fp8 KV cache | **T4 不支持**（需 SM 8.9+），本机才有这一臂 |
| 量化臂的显存对比 | **无数据**（未采 GPU 遥测）|
| AWQ 输出 sanity check | **未做**——`--include-text` 已传，产物里有 `text` 字段，但本次未逐条比对 |

分析见 `docs/benchmark_report.md` Part II 第 20 节。