# Kaggle T4 ×2 Runbook（历史文档）

> ⚠️ **本文档已归档为历史版本，请以
> [`runbook_three_models.md`](runbook_three_models.md) 第 5 节（Kaggle 1.5B 全流程）
> 和第 6 节（Kaggle 7B）为准。**
>
> 保留它的原因：这是项目最早的一版 Kaggle 笔记，记录了**实际撞到六个环境问题之前**的
> 设想。把它与实测结果对照，能看出哪些预设是错的——下面每一节都标注了**实测更正**。
>
> **主要过时之处**：本文档没有提到 T4 不支持 vLLM V1（必须退回 V0）、
> 没有 `--dtype half`、没有 transformers 降级、没有 TensorFlow 冲突，
> 而且第 4 节的 server 启动方式**在 Kaggle 上会失败**。

## 1. 上传与安装

将项目上传到 Kaggle Notebook 或从仓库 clone，然后在 Notebook 单元执行：

```bash
!pip install -r vllm-benchmark-lab/requirements.txt
!python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Kaggle 的 PyTorch/CUDA/vLLM 版本可能不同。先保存版本信息，再决定是否调整实验配置。

> **实测更正（2026-09-27）**：装依赖需要三条命令，见 `runbook_three_models.md` 第 5.1 节 Cell 3：
> 1. **卸载 TensorFlow**（与 vLLM 的 protobuf 要求冲突）；
> 2. **`transformers` 降到 4.51.1**（Kaggle 预装 5.0.0，与 vLLM 0.8.5.post1 不兼容）；
> 3. 再装 `vllm==0.8.5.post1`。
>
> 实测最终版本：Python 3.12.13 / torch 2.6.0+cu124 / CUDA 12.4 / vLLM 0.8.5.post1。

## 2. 模型缓存

模型首次加载会下载权重。推荐设置独立缓存目录，并在 Notebook 重启后复用 Dataset 或持久化输出：

```python
import os
os.environ["HF_HOME"] = "/kaggle/temp/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf-cache"
```

> **实测确认**：放 `/kaggle/temp` 而不是 `/kaggle/working` 是对的——
> `/kaggle/working` 有约 20GB 输出配额，权重放进去会挤掉结果文件的保存空间。
>
> ⚠️ **本文档缺了一条关键环境变量**：`VLLM_USE_V1`。Kaggle 上**必须**设成 `"0"`，
> 且要早于任何 `import vllm`——T4 不支持 V1，见 `runbook_three_models.md` 坑 8。

## 3. 单卡与双卡模式

单卡：

```bash
!CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py --config configs/qwen_1.5b.yaml
```

Tensor Parallel：将配置中的 `tensor_parallel_size` 改为 2，并确认 vLLM 版本和模型支持：

```bash
!CUDA_VISIBLE_DEVICES=0,1 python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tensor-parallel-size 2 \
  --requests 16 --max-tokens 128 \
  --output results/raw/kaggle-tp2.jsonl
```

两个单卡实例：分别绑定 GPU 0 和 GPU 1。此模式用于吞吐/隔离实验，不应与 Tensor Parallel 结果混称。

> **实测更正（2026-09-27）**：上面两条命令都缺 `VLLM_USE_V1=0` 和 `--dtype half`，
> 在 T4 上会失败或跑出不可用的配置。实际执行的命令见
> `runbook_three_models.md` 第 5.2 节 E7。
>
> **TP=2 的实测结果推翻了当时的预设**：T4 之间只有 PCIe、没有 NVLink，
> 当时预期"TP=2 很可能比 TP=1 更慢"，**实测 TP=2 快 24.8%**。
> 原因是 1.5B 太小、decode 受显存带宽限制，TP=2 把每卡要读的权重减半。
> 但**每卡均摊吞吐掉了 37.6%**——详见报告 Part II 第 21 节。
>
> **"两个单卡实例"（E7b）从未执行**，所以"独立实例是否优于 TP=2"仍是推算。

## 4. HTTP server 并发实验

在 Notebook 后台启动 server，并记录 PID、端口和启动参数：

```bash
!CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 > /kaggle/working/vllm-server.log 2>&1 &
```

建议先用 `curl` 确认服务，再运行 HTTP client。Notebook session 中断后，旧进程可能仍在运行，需要先检查端口和 PID。

> ❌ **实测更正（2026-09-27）：这一节的做法在 Kaggle 上不成立。**
>
> **Kaggle Notebook 不允许跨 cell 常驻的后台进程。** 用 `&`（或 `nohup ... &`）起的 server
> 会在 cell 执行结束后被回收，压测时端口上没有任何进程监听，
> 64 个请求会全部以 `APIConnectionError: Connection error.` 失败。
>
> 实测产物（**保留为失败实验，不删**）：
>
> ```text
> results/raw/kaggle-1.5b-http-concurrency-16.jsonl
> status=failed   ok=0   failed=64   output_tokens=0
> ```
>
> **后果**：Kaggle 侧**零 per-request 指标**（TTFT / TPOT / p95 / p99 / req/s），
> 导致 prefix caching 与 chunked prefill 两个实验的**尾延迟问题在 T4 上同样没答**，
> 也做不了 server 与离线两条路径的交叉验证。
>
> **✅ 正确做法**：把 server 的整个生命周期**关在一个 cell 内**
> （`subprocess.Popen` → 轮询 `/v1/models` 直到就绪 → 压测 → `terminate()`）。
> 完整代码见 `runbook_three_models.md` 第 5.3 节。**关键是不要跨 cell。**
> 这条路径**本身尚未验证**，是当前的最高优先级待办之一。

## 5. GPU 监控

另一个单元启动采集器：

```bash
!python scripts/collect_gpu_metrics.py \
  --run-id kaggle-20260925-baseline \
  --output results/raw/kaggle-gpu.jsonl \
  --duration 180 --interval 1
```

> **实测更正（2026-09-27）**：1.5B 全矩阵那一轮**一个 GPU 遥测文件都没有采集**。
> 后果：并发实验的"离显存上限还有多远"、量化实验的"省了多少显存"**全部无解**。
> 报告 Part II 第 20.4 节只能给出**按位宽推算**的区间，明确标注它不是测量值。
>
> **下一轮务必先起采集器再跑 benchmark**——它是唯一能回答显存问题的东西。

## 6. 结果导出

实验完成后下载以下目录：

- `results/raw/`
- `results/processed/`
- `results/plots/`
- `docs/benchmark_report.md`
- `vllm-server.log`

报告中必须记录：GPU 型号、数量、驱动/CUDA、Python、PyTorch、vLLM、模型 revision、并行模式和失败实验。

> **实测更正（2026-09-27）**：光下载 `results/raw/` **不够**。那一轮 22 个 JSONL 里
> **缺三个关键字段**（`VLLM_USE_V1`、`transformers` 版本、attention backend），
> 导致"全部跑在 V0"这一前提**只能靠操作者陈述**。
>
> **`environment_metadata()` 已修补**（2026-09-27，新增 `env_vars` 与 `transformers`），
> 但**已有产物无效**。另外 `results/raw/env-kaggle-1.5b.json` 记的是降级前的
> `transformers: 5.0.0`，与实际用的 4.51.1 矛盾，**该快照不可信、需要重拍**。
>
> **下一轮还要额外汇出 `vllm-server.log`**：日志里的 `enable_prefix_caching`、
> attention backend、`Using V0/V1 engine` 三行，**恰好就是产物里缺的那三个字段**。
