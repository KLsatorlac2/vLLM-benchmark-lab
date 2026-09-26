# Kaggle T4 x2 Runbook

## 1. 上传与安装

将项目上传到 Kaggle Notebook 或从仓库 clone，然后在 Notebook 单元执行：

```bash
!pip install -r vllm-benchmark-lab/requirements.txt
!python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Kaggle 的 PyTorch/CUDA/vLLM 版本可能不同。先保存版本信息，再决定是否调整实验配置。

## 2. 模型缓存

模型首次加载会下载权重。推荐设置独立缓存目录，并在 Notebook 重启后复用 Dataset 或持久化输出：

```python
import os
os.environ["HF_HOME"] = "/kaggle/working/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/working/hf-cache"
```

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

## 4. HTTP server 并发实验

在 Notebook 后台启动 server，并记录 PID、端口和启动参数：

```bash
!CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 > /kaggle/working/vllm-server.log 2>&1 &
```

建议先用 `curl` 确认服务，再运行 HTTP client。Notebook session 中断后，旧进程可能仍在运行，需要先检查端口和 PID。

## 5. GPU 监控

另一个单元启动采集器：

```bash
!python scripts/collect_gpu_metrics.py \
  --run-id kaggle-20260925-baseline \
  --output results/raw/kaggle-gpu.jsonl \
  --duration 180 --interval 1
```

## 6. 结果导出

实验完成后下载以下目录：

- `results/raw/`
- `results/processed/`
- `results/plots/`
- `docs/benchmark_report.md`
- `vllm-server.log`

报告中必须记录：GPU 型号、数量、驱动/CUDA、Python、PyTorch、vLLM、模型 revision、并行模式和失败实验。
