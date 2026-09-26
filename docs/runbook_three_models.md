# 三模型实验操作手册

目标：在三个模型上跑完整套推理实验，产出一份可回溯的报告。

| 模型 | 在哪跑 | 为什么 |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 本机 RTX 4060 8GB | 全矩阵（含量化、fp8 KV cache、16384 上下文）|
| Qwen2.5-1.5B-Instruct | Kaggle T4 16GB ×2 | 全矩阵 + TP=2 对比 + 高并发 |
| Qwen2.5-7B-Instruct（AWQ/GPTQ）| Kaggle T4 16GB ×2 | 量化、TP、上下文上限；FP16 单卡放不下 |

这份手册只写"怎么操作"。口径定义仍以 `project_summary.md` 和 `README.md` 为准。

---

## 0. 本机实测确认的事实

以下是 2026-09-25 在本机实测（vLLM 0.8.5.post1）确认的日志，后面的坑都基于这些事实：

```text
Initializing a V1 LLM engine (v0.8.5.post1) ...
    dtype=torch.bfloat16, max_seq_len=1024, enable_prefix_caching=False,
    chunked_prefill_enabled=True
Chunked prefill is enabled with max_num_batched_tokens=2048.
Using Flash Attention backend on V1 engine.
GPU KV cache size: 125,680 tokens          # gpu-memory-utilization=0.45, max-model-len=1024
torch.compile takes 51.11 s in total       # 首次启动编译耗时
```

结论：

1. **引擎是 V1**。
2. **chunked prefill 在 V1 下恒为 True**，无法关闭（见坑 1）。
3. 显式传 `enable_prefix_caching=False` 生效；server 端 `--no-enable-prefix-caching` 参数被接受（见坑 2）。
4. 本机 4060（Ada）`dtype=auto` 解析为 **bfloat16**。
5. server 首次启动约 2 分钟（torch.compile 占 ~51s），热启动约 83 秒。

2026-09-26 跑完 0.5B 全矩阵（E1–E6）时又实测到两个 V1 的限制：

6. **V0 下 `max_num_batched_tokens < max_model_len` 会启动即崩**（`ValueError`），E5 的 `max_model_len` 因此从 4608 降到 4096（见坑 6）。
7. **V1 没有实现 fp8 KV cache**（`NotImplementedError`），E6 的 fp8 KV 臂必须用 `VLLM_USE_V1=0` 走 V0（见坑 7）。

> 第 6、7 条都说明同一件事：**V1 相对 V0 缺少若干特性**。本机默认 V1，所以
> 凡是要用到这些特性的臂，都得显式退回 V0，并在报告里标注引擎版本。

---

## 1. 七个坑：五个会静默污染结果，两个会直接报错

坑 1–5 的共同点是**命令能跑完、结果却不可信**，必须靠人去看日志和数字才能发现。
坑 6–7 是 2026-09-26 本机跑 0.5B 全矩阵时实际撞到的两个报错，**会在启动阶段直接崩**，反而容易发现；
之所以还是记下来，是因为它们各自改动了实验的配置口径，报告里必须交代（见坑 6、坑 7 末尾的"对结果的影响"）。

### 坑 1：V1 下 chunked prefill 关不掉

上面日志已经证明 V1 里 `chunked_prefill_enabled=True` 是强制值。传 `--enable-chunked-prefill` 开关没有意义，`--no-enable-chunked-prefill` 甚至不被识别。

要做真实的开/关 A/B，必须退回 V0 引擎：

```bash
VLLM_USE_V1=0 <命令>
```

**注意**：V0 在更新版本的 vLLM 里已被删除，`VLLM_USE_V1=0` 会失效（0.15.0 起）。本机的 0.8.5.post1 仍然有效。所以：

- chunked prefill 的结论必须标注"V0 引擎"。
- 同时用 V1 默认配置跑一次作为参考基线，报告里分开写。

### 坑 2：V1 下 prefix caching 默认是**开**的

`class: LLM` 和 server 在 V1 下默认 `enable_prefix_caching=True`。

- `scripts/run_vllm.py` 显式传入 `enable_prefix_caching=args.enable_prefix_caching`（False），所以**离线 runner 是安全的**。
- 但 **server 模式默认开着**。`python -m vllm.entrypoints.openai.api_server` 不加参数就是开。必须显式：

```bash
--no-enable-prefix-caching
```

### 坑 3：`load_prompts` 会把 prompt 文件循环填充到 `--requests`

`scripts/common.py` 里：

```python
prompts = (prompts * ((count + len(prompts) - 1) // len(prompts)))[:count]
```

文件里 4 条 prompt、`--requests 16`，实际会变成"这 4 条各重复 4 次"。**一旦 prefix caching 是开的，第 2 轮开始全部命中缓存**，吞吐会虚高。

规则：**prompt 文件的条数 == `--requests`**。本手册生成的所有 prompt 文件都按这条规则配好了。

### 坑 4：`prompt_len + max_tokens` 必须 ≤ `max_model_len`

4096 token 的 prompt 配 `--max-model-len 4096 --max-tokens 64` 会被 vLLM 直接拒绝。上下文实验一律按"prompt 长度 + 输出 + 余量"设置：

| prompt 实际长度 | 建议 `--max-model-len` |
| --- | --- |
| 512 | 1024 |
| 2048 | 2560 |
| 4096 | 4608 |
| 8192 | 8704 |
| 16384 | 16896 |

> 上表是**上限**，只在 V1 下可以直接用。**V0 下还多一条约束**：
> `max_model_len` 不能超过 `max_num_batched_tokens`，否则启动即崩（见坑 6）。
> E5 就是撞在这条上，最后三个臂统一取了 4096。

### 坑 5：离线 `LLM.generate` 只有 batch 级数字

`run_vllm.py` 测的是整个 batch 的耗时，同一 batch 内每条记录数值相同。所以：

- 离线结果只能讲 **batch latency / batch throughput**，不能讲 p95、TTFT、TPOT。
- 需要 **per-request TTFT / TPOT / p95** 时，用新增的 `scripts/benchmark_http.py` 走 server 模式。

`scripts/summarize_results.py` 会自动给每个 run 打上 `latency_mode=batch` 或 `per-request` 标签，并对 batch run 隐藏无意义的百分位。

### 坑 6：V0 下 `max_num_batched_tokens` 必须 ≥ `max_model_len`（实测报错）

**报错原文**（2026-09-26，本机 0.5B，E5 chunked prefill 的 OFF 臂）：

```text
ValueError: max_num_batched_tokens (4096) is smaller than max_model_len (4608).
This effectively limits the maximum sequence length to max_num_batched_tokens
and makes vLLM reject longer sequences.
Please increase max_num_batched_tokens or decrease max_model_len.
```

**触发条件**：按本手册第 4 节 E5 最初写的参数跑 OFF 臂——

```bash
--max-model-len 4608 --max-num-batched-tokens 4096      # ← 崩
```

**原因**：坑 1 已经说明 V0 没有 chunked prefill，一条序列必须能在**一个 scheduler step** 里整个放下。step 的 token 预算是 `max_num_batched_tokens`，它比 `max_model_len` 小的时候，vLLM 无法保证任何达到 `max_model_len` 的序列能被调度，于是直接在启动时拒绝，而不是等到运行时才 OOM。V1 不受这条约束（chunked prefill 恒开，长 prefill 会被自动切块），所以同一个 `2048 / 4608` 组合在 V1 参考臂上是能跑的。

**修法**：把 E5 三个臂的 `--max-model-len` 一律降到 **4096**（`local-0.5b-chunked-{off,on,v1ref}.jsonl` 都是这个值），OFF 臂的 4096 预算正好能放下"3584 长 prompt + 64 输出 = 3648"。

**对结果的影响**：E5 的上下文上限从 4608 降到 4096，但混合 workload 里最长的一条只有 3584 token，**实际用到的长度没有变**，`input_tokens` 仍然是 `1824(3584)`。三个臂的 `max_model_len` 完全一致，所以 OFF/ON/V1ref 的对比依然干净。报告第 9 节要写明这个降级，不要写成 4608。

> 检查规则：**V0 下 `max_num_batched_tokens ≥ max_model_len`**。做 chunked prefill A/B 时，
> OFF 臂的预算要 ≥ 最长 prompt + 输出，ON 臂则可以故意压小。

### 坑 7：V1 不支持 fp8 KV cache，必须退回 V0（实测报错）

**报错原文**（2026-09-26，本机 0.5B，E6 fp8 KV cache 臂，`VLLM_USE_V1=1`）：

```text
NotImplementedError: VLLM_USE_V1=1 is not supported with --kv-cache-dtype
```

**原因**：本机默认跑在 V1 引擎上（第 0 节已确认），而 V1 在 0.8.5.post1 里**没有实现 fp8 KV cache**，只接受了参数名、没实现后端，于是直接抛 `NotImplementedError`。这与坑 1 是同一个根因——**V1 相对 V0 砍掉的特性不止一个**。

**修法**：fp8 KV cache 这一臂显式退回 V0：

```bash
VLLM_USE_V1=0 $PY scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --kv-cache-dtype fp8 \
  ...
```

`results/raw/local-0.5b-kvfp8.jsonl` 就是这么跑出来的。

**对结果的影响**：`local-0.5b-kvfp8.jsonl` 是 **V0 引擎**的数字，而它的对照基线 `local-0.5b-vllm-batch16.jsonl`（同一份默认 prompt、16 请求 × 64 输出）是 **V1** 跑的。两者的差距里同时混了"引擎版本"（V0 vs V1）和"KV dtype"两件事：

| run | 引擎 | `gpu_memory_utilization` | batch 延迟 | batch 吞吐 |
| --- | --- | --- | --- | --- |
| `vllm-batch16`（FP16 基线）| V1 | 0.90 | 952.11 ms | 1075.51 tok/s |
| `kvfp8` | **V0** | 0.75 | 1094.47 ms | 935.61 tok/s |

吞吐差约 **13%**，但**不能**归因给 fp8——引擎版本本身就足以解释这个量级。所以报告第 10 节只能写成"**fp8 KV cache 在本机 0.5B 上没有可观测的吞吐收益**"，并标注该臂跑在 V0 上；绝不能写成"fp8 比 fp16 慢 13%"。`gpu_memory_utilization` 也不同（0.90 vs 0.75），是第二个需要交代的差异。

> 精确归因的做法（本次未做，留给后续）：把 FP16 基线也用 `VLLM_USE_V1=0`、
> `gpu_memory_utilization=0.75` 重跑一次，做同引擎、同显存配额的对照。

**暴露出的记录缺陷：`VLLM_USE_V1` 没有被写进 JSONL。**

`scripts/common.py` 的 `environment_metadata()` 只记了 python/torch/cuda/gpu/vllm，**没有记 `VLLM_USE_V1`**。
所以事后光看 `results/raw/*.jsonl`，**无法判断某个 run 到底跑在 V0 还是 V1 上**——坑 1、6、7 都依赖这个变量，而它恰恰是唯一没被记下来的那个。

本次的后果：`local-0.5b-awq.jsonl`（932.99 tok/s）的引擎版本**无法从产物中确认**。
它的吞吐和 `local-0.5b-kvfp8.jsonl`（935.61 tok/s，已知 V0）几乎相同，也和 FP16 V1 基线（1075.51 tok/s）差约 13%，
但仅凭这些数字无法区分"AWQ 在 V1 上就是这个速度"和"AWQ 这一臂其实也跑在 V0 上"。

**修法（已列入待办）**：在 `environment_metadata()` 里补 `VLLM_USE_V1` / `VLLM_WORKER_MULTIPROC_METHOD` 等关键环境变量，
或者让 `run_vllm.py` 把 `llm.llm_engine.vllm_config` 里的引擎版本写进 metadata 行。**补上之前，任何涉及引擎切换的实验都要在报告里人工标注。**

> 顺带一条纪律：**AWQ 臂与 fp8 KV 臂的引擎版本可能不同，两者不要直接互比**；
> 它们和 FP16 基线之间也各自存在引擎/显存配额差异。量化一节（报告第 10 节）只写同引擎内可比的结论。

### 额外两条硬件限制（Kaggle T4）

- **T4 不支持 bfloat16**（Turing 无 BF16）。T4 上 `dtype=auto` 会解析成 float16，**不要手动指定 bfloat16**。
- **T4 不支持 fp8 KV cache**（需要 SM 8.9+）。`--kv-cache-dtype fp8` 只在本机 4060（Ada）上做。
- T4 上 AWQ/GPTQ 没有 Marlin kernel（需要 SM 80+），走较慢的解量化路径。所以 **T4 上量化的收益主要是显存，不是速度**，这本身就是一条值得写的结论。

---

## 2. 实验矩阵

| 编号 | 实验 | 0.5B 本机 | 1.5B Kaggle | 7B Kaggle |
| --- | --- | --- | --- | --- |
| E1 | HF vs vLLM 基线 | ✅ | ✅ | ❌ 单卡放不下 FP16 |
| E2 | 并发扫描 | ✅ 16/32/64 | ✅ 16/64/128/256 | ✅ 16/64 |
| E3 | 上下文长度 | ✅ 512→16384 | ✅ 512→16384 | ✅ 512→8192 |
| E4 | Prefix caching | ✅ | ✅ | ✅ |
| E5 | Chunked prefill（V0）| ✅ | ✅ | ✅ |
| E6 | 量化 | ✅ AWQ + fp8 KV | ✅ AWQ | ✅ AWQ vs GPTQ |
| E7 | 并行 | ❌ 单卡 | ✅ TP=1 vs TP=2 | ✅ TP=1 vs TP=2 |

---

## 3. 通用准备

### 3.1 本机环境

```bash
cd ~/vllm-benchmark-lab
export PY=/home/asus/venvs/vllm/bin/python
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1          # 默认；跑 chunked prefill 的 A/B 时改成 0
$PY -c "import torch, vllm, transformers; print(torch.__version__, torch.cuda.device_count(), vllm.__version__)"
```

### 3.2 生成 prompt 文件（只做一次）

新脚本 `scripts/make_prompts.py` 生成**精确 token 长度**的 prompt，并在结束时逐条复核实际长度。

四种模式：

| 模式 | 用途 |
| --- | --- |
| `distinct` | 每条 prompt 长度精确、**开头各不相同**，没有共享前缀。用于上下文长度、并发实验 |
| `shared-prefix` | 所有 prompt 开头有一段**逐字节相同**的前缀，之后分叉。用于 prefix caching 的"开" |
| `random-prefix` | 每条 prompt 有各自唯一的前缀，尾部相同。是 `shared-prefix` 的**对照组** |
| `mixed` | 长、短 prompt 交替。用于 chunked prefill |

批量生成：

```bash
cd ~/vllm-benchmark-lab

# E3 上下文长度：每组总 prefill token 数固定为 8192，只改单条长度
$PY scripts/make_prompts.py --mode distinct --target-tokens 512   --num-prompts 16 --output experiments/context_length/prompts_512.txt   --strict
$PY scripts/make_prompts.py --mode distinct --target-tokens 2048  --num-prompts 4  --output experiments/context_length/prompts_2048.txt  --strict
$PY scripts/make_prompts.py --mode distinct --target-tokens 4096  --num-prompts 2  --output experiments/context_length/prompts_4096.txt  --strict
$PY scripts/make_prompts.py --mode distinct --target-tokens 8192  --num-prompts 1  --output experiments/context_length/prompts_8192.txt  --strict
$PY scripts/make_prompts.py --mode distinct --target-tokens 16384 --num-prompts 1 --output experiments/context_length/prompts_16384.txt --strict

# E2 并发：固定 512 token，64 条互不相同
$PY scripts/make_prompts.py --mode distinct --target-tokens 512 --num-prompts 64 --output experiments/concurrency/prompts_512x64.txt --strict

# E4 prefix caching：共享前缀 1024 + 各自不同的尾巴 128
$PY scripts/make_prompts.py --mode shared-prefix --prefix-tokens 1024 --target-tokens 1152 --num-prompts 16 --output experiments/prefix_caching/prompts_shared_1024.txt --strict
$PY scripts/make_prompts.py --mode random-prefix --prefix-tokens 1024 --target-tokens 1152 --num-prompts 16 --output experiments/prefix_caching/prompts_random_1024.txt --strict

# E5 chunked prefill：8 条长（3584）+ 8 条短（64）交替
$PY scripts/make_prompts.py --mode mixed --long-tokens 3584 --short-tokens 64 --num-prompts 16 --output experiments/chunked_prefill/prompts_mixed.txt
```

每条命令都会打印实际 token 数和一份 `*.manifest.json`（记录 workload 分布，填报告第 3 节用）。`--strict` 保证长度精确，不精确会以非零码退出。

**为什么上下文实验要让总 token 数固定**：512×16、2048×4、4096×2、8192×1 四组的总 prefill 工作量都是 8192 token。这样 batch latency 的变化只来自**单条序列变长**，而不是"总工作量变多"。否则长度和总量两个变量会混在一起。

### 3.3 结果命名规范

```text
results/raw/<环境>-<模型>-<实验>-<设置>.jsonl
```

例如 `local-0.5b-concurrency-16.jsonl`、`kaggle-1.5b-context-4096.jsonl`。环境前缀保证三台机器的文件不会互相覆盖。

已有的 `hf-0.5b.jsonl`、`vllm-0.5b.jsonl`、`concurrency-16.jsonl`、`concurrency-64.jsonl` 会被下面的命令用新命名覆盖/取代，可以忽略。

---

## 4. 本机 0.5B 全流程

### E1 HF vs vLLM

**注意公平性**：`run_hf.py` 是逐条串行（batch=1），而 `run_vllm.py` 是整批并行。直接拿 `requests=4` 的 HF 对比 `requests=4` 的 vLLM，差异里混了"引擎"和"批处理"两件事。拆成两步：

```bash
# E1a 单请求对单请求：隔离引擎差异
$PY scripts/run_hf.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 1 --max-tokens 64 \
  --output results/raw/local-0.5b-hf-single.jsonl

$PY scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 1 --max-num-seqs 1 --max-model-len 1024 --max-tokens 64 \
  --output results/raw/local-0.5b-vllm-single.jsonl

# E1b 批处理收益：vLLM 自己从 1 条加到 16 条
$PY scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --requests 16 --max-num-seqs 16 --max-model-len 1024 --max-tokens 64 \
  --output results/raw/local-0.5b-vllm-batch16.jsonl
```

观察点：`requests=1` 时两个引擎的差距是纯引擎差距；`requests=1 → 16` 的差距是批处理收益。

### E2 并发扫描

```bash
for SEQ in 16 32 64; do
  $PY scripts/run_vllm.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompts experiments/concurrency/prompts_512x64.txt \
    --requests 64 --max-num-seqs $SEQ \
    --max-model-len 1024 --max-tokens 64 \
    --gpu-memory-utilization 0.75 \
    --output results/raw/local-0.5b-concurrency-$SEQ.jsonl
done
```

观察点：`max_num_seqs` 越大，batch throughput 越高，但 batch latency 也会变长；显存到顶时会出现 preemption 或 OOM。**OOM 的实验要保留**，在报告里写成失败实验。

### E3 上下文长度

```bash
run_ctx() {
  $PY scripts/run_vllm.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompts experiments/context_length/prompts_$1.txt \
    --requests $2 --max-num-seqs $2 \
    --max-model-len $3 --max-tokens 64 \
    --gpu-memory-utilization 0.75 \
    --output results/raw/local-0.5b-context-$1.jsonl
}
run_ctx 512   16 1024
run_ctx 2048  4  2560
run_ctx 4096  2  4608
run_ctx 8192  1  8704
run_ctx 16384 1  16896
```

0.5B 的 KV cache 只有约 12 KB/token，所以本机可以一路做到 16384。**这是本机相对 T4 的独特优势**：T4 上 7B 做 16384 要 917 MB KV，本机 0.5B 只要 196 MB。

跑完立刻核对真实长度（不要相信配置上限）：

```bash
$PY scripts/make_prompts.py --check experiments/context_length/prompts_*.txt
grep -o '"input_tokens":[0-9]*' results/raw/local-0.5b-context-8192.jsonl
```

### E4 Prefix caching

四组，两两对照：

```bash
for ARM in shared random; do
  for CACHE in off on; do
    FLAG=""
    [ "$CACHE" = "on" ] && FLAG="--enable-prefix-caching"
    $PY scripts/run_vllm.py \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --prompts experiments/prefix_caching/prompts_${ARM}_1024.txt \
      --requests 16 --max-num-seqs 16 \
      --max-model-len 1280 --max-tokens 64 \
      --gpu-memory-utilization 0.75 $FLAG \
      --output results/raw/local-0.5b-prefix-${ARM}-${CACHE}.jsonl
  done
done
```

**必须验证缓存真的关了**（坑 2）：对比前先在日志里确认。

```bash
$PY -c "import vllm; from vllm import LLM; LLM(model='Qwen/Qwen2.5-0.5B-Instruct', enable_prefix_caching=False, max_model_len=1024, gpu_memory_utilization=0.45)" 2>&1 | grep -i "prefix caching"
```

预期结论：

- `shared-on` 相对 `shared-off`：明显更快（1024 token 前缀全部命中）。
- `random-on` 相对 `random-off`：**基本没有变化**。如果这条也变快了，说明有别的共享前缀（例如 prompt 都以同一段文字开头），要回去检查生成参数。

`shared` 和 `random` 两臂总长度相同（1152），都是 16 条，所以可以干净地归因到"前缀是否共享"。

### E5 Chunked prefill（必须用 V0）

```bash
# OFF 臂：不切块，单步 token 预算必须能放下 3584 的长 prompt
# max-model-len 必须是 4096 而不是 4608：V0 下预算(4096) < max_model_len(4608) 会直接崩，见坑 6
VLLM_USE_V1=0 $PY scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts experiments/chunked_prefill/prompts_mixed.txt \
  --requests 16 --max-num-seqs 16 \
  --max-model-len 4096 --max-tokens 64 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.75 \
  --output results/raw/local-0.5b-chunked-off.jsonl

# ON 臂：预算压到 2048，长 prefill 被切成 2 块，短请求可以插进去
VLLM_USE_V1=0 $PY scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts experiments/chunked_prefill/prompts_mixed.txt \
  --requests 16 --max-num-seqs 16 \
  --max-model-len 4096 --max-tokens 64 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.75 \
  --output results/raw/local-0.5b-chunked-on.jsonl

# V1 参考臂：chunked prefill 恒开，做对照记录
VLLM_USE_V1=1 $PY scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts experiments/chunked_prefill/prompts_mixed.txt \
  --requests 16 --max-num-seqs 16 \
  --max-model-len 4096 --max-tokens 64 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.75 \
  --output results/raw/local-0.5b-chunked-v1ref.jsonl
```

> **`max-model-len` 三个臂统一取 4096**（不是手册早先写的 4608）。
> OFF 臂受坑 6 约束必须 ≤ `max-num-batched-tokens`；为了三个臂可比，ON 和 V1ref 也跟着取 4096。

两个臂的 `max-num-batched-tokens` 不同是**必然的**，不是污染：关掉切块时，单条 prompt 必须能在一个 step 里放下，所以预算必须 ≥ 3584。这个差异本身就是"切块 vs 不切块"的操作定义。报告里要写清楚。

观察点：短请求（64 token）的等待时间。离线模式所有请求同时入队，理论上开切块后短请求不必等长 prefill 全部完成。要拿到真正的短请求尾延迟，用第 4.7 节的 server 模式。

### E6 量化（本机独占）

```bash
# AWQ 0.5B
$PY scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct-AWQ \
  --quantization awq \
  --requests 16 --max-num-seqs 16 --max-model-len 1024 --max-tokens 64 \
  --gpu-memory-utilization 0.75 \
  --output results/raw/local-0.5b-awq.jsonl --include-text

# fp8 KV cache（Ada 支持，T4 不支持）
# 必须 VLLM_USE_V1=0：V1 未实现 fp8 KV cache，会抛 NotImplementedError，见坑 7
VLLM_USE_V1=0 $PY scripts/run_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --kv-cache-dtype fp8 \
  --requests 16 --max-num-seqs 16 --max-model-len 1024 --max-tokens 64 \
  --gpu-memory-utilization 0.75 \
  --output results/raw/local-0.5b-kvfp8.jsonl
```

> **注意 fp8 KV 臂跑在 V0 上**，而 FP16 基线跑在 V1 上，两者不是同引擎对照（见坑 7）。
> 报告里必须标注引擎版本；想要严格归因，基线要加 `VLLM_USE_V1=0` 重跑。

**FP16 基线不能直接用 E2 的 `concurrency-16`。** 上面 AWQ / fp8 KV 两条命令都没有传 `--prompts`，
所以它们用的是 `common.py` 里那 4 条默认 prompt（实测 `input_tokens=10~12`）、16 请求；
而 `concurrency-16` 传的是 `prompts_512x64.txt`（512 token × 64 请求）。
两者的 prompt 长度差 50 倍，**batch 延迟根本不可比**。

同一份默认 prompt 的 FP16 基线是 E1b 的 `local-0.5b-vllm-batch16.jsonl`，但它的
`--gpu-memory-utilization` 是 0.90 而量化臂是 0.75，**仍然不是完全对等**。
要写量化结论，建议用同 prompt、同 `gpu-memory-utilization` 重跑一条 FP16 基线。

**checkpoint 必须先验证能加载**（项目规则）。上面命令如果报 `Cannot find model` 或量化 kernel 不支持，就换成同名仓库确认后重跑；不支持的组合按失败实验保留。

**量化必须做输出 sanity check**：用 `--include-text` 生成的文本和 FP16 基线对比，确认不是乱码。`--include-text` 会显著增大 JSONL，比较完可以把文本字段删掉再归档。

### E7 单卡说明

本机只有 1 张卡，`--tensor-parallel-size` 固定 1，E7 跳过。

### 4.7 Server 模式（拿真实 TTFT / TPOT / p95）

E2/E3 的离线数字只有 batch 级。要写 p95 TTFT、per-request TPOT，用两个新脚本。

**终端 1：启动 server**（首次约 2 分钟，torch.compile 占 ~51s）

```bash
cd ~/vllm-benchmark-lab
CUDA_VISIBLE_DEVICES=0 /home/asus/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen0.5b \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 1024 --max-num-seqs 16 \
  --gpu-memory-utilization 0.6 \
  --no-enable-prefix-caching
```

**终端 2：确认服务可用，再压测**

```bash
curl -s http://127.0.0.1:8000/v1/models | head -c 300

cd ~/vllm-benchmark-lab
$PY scripts/benchmark_http.py \
  --model qwen0.5b \
  --prompts experiments/concurrency/prompts_512x64.txt \
  --requests 64 --concurrency 16 --max-tokens 64 \
  --output results/raw/local-0.5b-http-concurrency-16.jsonl
```

`--concurrency` 是同时在线的请求数，语义对应 `max_num_seqs`：设成 64 = 一次性全发（接近离线 burst），设成 16 = 有排队。

`benchmark_http.py` 会输出 per-request 的 `ttft_ms` / `tpot_ms` / `latency_ms`，以及 mean/p50/p95/p99。这组数字才能和 `max_num_seqs` 的调度效果对应起来。

**终端 3（可选）：GPU 采集**

```bash
$PY scripts/collect_gpu_metrics.py \
  --run-id local-0.5b-http-concurrency-16 \
  --output results/raw/gpu-local-0.5b-http-16.jsonl \
  --interval 0.5 --duration 120
```

---

## 5. Kaggle 1.5B 全流程

### 5.1 上传与安装

推荐的传递方式：把整个仓库打包成私有的 Kaggle Dataset，挂载后复制到 `/kaggle/working`。

**Cell 1（必须在任何 import 之前）**

```python
import os
os.environ["HF_HOME"] = "/kaggle/temp/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/kaggle/temp/hf-cache"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"          # chunked prefill 的 A/B 再单独改成 "0"
```

> Notebook 里设置环境变量必须早于 `import transformers/vllm`，否则不生效。
>
> 模型缓存放 `/kaggle/temp` 而不是 `/kaggle/working`：`/kaggle/working` 有约 20GB 输出配额，权重放进去会挤掉结果文件的保存空间。

**Cell 2：确认 GPU 与版本**

```python
!nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
```

**Cell 3：安装 vLLM**

```python
!pip install -q vllm==0.8.5.post1
```

Kaggle 预装的 torch 版本和 vLLM 0.8.5.post1 可能不匹配，pip 可能连带升级/降级 torch。**这是可以接受的，但必须记录**——项目的规则本来就要求跨硬件结果分别标注。

**Cell 4：记录本次环境到结果目录**

```python
!python -c "import torch, vllm, transformers, json, os; \
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, \
'vllm': vllm.__version__, 'transformers': transformers.__version__, \
'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, indent=2))" \
> results/raw/env-kaggle-1.5b.json
```

**Cell 5：进入项目目录**

```python
!cp -r /kaggle/input/vllm-benchmark-lab /kaggle/working/vllm-benchmark-lab
%cd /kaggle/working/vllm-benchmark-lab
!ls scripts/
```

**Cell 6：验证 prompt 文件在 Kaggle 的 tokenizer 下长度一致**

```python
!python scripts/make_prompts.py --check experiments/context_length/prompts_512.txt \
  experiments/context_length/prompts_4096.txt
```

Qwen2.5 各尺寸共用同一套 tokenizer，所以本机生成的 prompt 文件可以直接用。这一步是**验证**，不是重新生成——如果长度对不上，就以 Kaggle 上实测的 `input_tokens` 为准并在报告里说明。

> Internet：Kaggle Notebook 默认可能关闭外网，下载 HF 权重需要打开 Internet 开关，或者把模型作为 Dataset 挂载。

### 5.2 实验命令（Kaggle cell 里用 `!` 前缀）

**E1 HF vs vLLM**

```bash
!python scripts/run_hf.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --requests 1 --max-tokens 64 --output results/raw/kaggle-1.5b-hf-single.jsonl

!python scripts/run_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --requests 1 --max-num-seqs 1 --max-model-len 1024 --max-tokens 64 \
  --gpu-memory-utilization 0.85 --output results/raw/kaggle-1.5b-vllm-single.jsonl

!python scripts/run_vllm.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --requests 16 --max-num-seqs 16 --max-model-len 1024 --max-tokens 64 \
  --gpu-memory-utilization 0.85 --output results/raw/kaggle-1.5b-vllm-batch16.jsonl
```

**E2 并发扫描（单卡）**

```bash
!for SEQ in 16 64 128 256; do \
  CUDA_VISIBLE_DEVICES=0 python scripts/run_vllm.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --prompts experiments/concurrency/prompts_512x64.txt \
    --requests 64 --max-num-seqs $SEQ \
    --max-model-len 1024 --max-tokens 64 \
    --gpu-memory-utilization 0.85 \
    --output results/raw/kaggle-1.5b-concurrency-$SEQ.jsonl; \
done
```

T4 16GB 跑 1.5B：权重约 3.1GB，KV cache 约 28 KB/token。`gpu_memory_utilization=0.85` 时 KV 池约 10GB ≈ 36 万 token，`max_num_seqs=256` 配 1024 上下文约需 26 万 token，放得下。

**E3 上下文长度**

```bash
!for CFG in "512 16 1024" "2048 4 2560" "4096 2 4608" "8192 1 8704" "16384 1 16896"; do \
  set -- $CFG; \
  CUDA_VISIBLE_DEVICES=0 python scripts/run_vllm.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --prompts experiments/context_length/prompts_$1.txt \
    --requests $2 --max-num-seqs $2 --max-model-len $3 --max-tokens 64 \
    --gpu-memory-utilization 0.85 \
    --output results/raw/kaggle-1.5b-context-$1.jsonl; \
done
```

**E4 Prefix caching**

```bash
!for ARM in shared random; do for CACHE in off on; do \
  FLAG=""; [ "$CACHE" = "on" ] && FLAG="--enable-prefix-caching"; \
  CUDA_VISIBLE_DEVICES=0 python scripts/run_vllm.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --prompts experiments/prefix_caching/prompts_${ARM}_1024.txt \
    --requests 16 --max-num-seqs 16 --max-model-len 1280 --max-tokens 64 \
    --gpu-memory-utilization 0.85 $FLAG \
    --output results/raw/kaggle-1.5b-prefix-${ARM}-${CACHE}.jsonl; \
done; done
```

**E5 Chunked prefill（V0）**

```bash
!CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompts experiments/chunked_prefill/prompts_mixed.txt \
  --requests 16 --max-num-seqs 16 --max-model-len 4096 --max-tokens 64 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 \
  --output results/raw/kaggle-1.5b-chunked-off.jsonl

!CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompts experiments/chunked_prefill/prompts_mixed.txt \
  --requests 16 --max-num-seqs 16 --max-model-len 4096 --max-tokens 64 \
  --max-num-batched-tokens 2048 --enable-chunked-prefill \
  --gpu-memory-utilization 0.85 \
  --output results/raw/kaggle-1.5b-chunked-on.jsonl
```

> `--max-model-len` 用 **4096**（不是 4608）：与本机一致，避开坑 6。
> T4 不支持 fp8 KV cache，所以 Kaggle 上没有坑 7 对应的那一臂。

**E6 量化**

```bash
!CUDA_VISIBLE_DEVICES=0 python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct-AWQ --quantization awq \
  --requests 16 --max-num-seqs 16 --max-model-len 1024 --max-tokens 64 \
  --gpu-memory-utilization 0.85 \
  --output results/raw/kaggle-1.5b-awq.jsonl --include-text
```

**E7 TP=1 vs TP=2**

```bash
# 单卡基线（前面已有 concurrency-16，这里用同一份 prompt 保持可比）
!CUDA_VISIBLE_DEVICES=0 python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompts experiments/concurrency/prompts_512x64.txt \
  --requests 64 --max-num-seqs 64 --max-model-len 1024 --max-tokens 64 \
  --gpu-memory-utilization 0.85 \
  --output results/raw/kaggle-1.5b-tp1.jsonl

# 双卡 Tensor Parallel
!CUDA_VISIBLE_DEVICES=0,1 python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tensor-parallel-size 2 \
  --prompts experiments/concurrency/prompts_512x64.txt \
  --requests 64 --max-num-seqs 64 --max-model-len 1024 --max-tokens 64 \
  --gpu-memory-utilization 0.85 \
  --output results/raw/kaggle-1.5b-tp2.jsonl
```

T4 之间走 PCIe、没有 NVLink，TP=2 的通信开销可能吃掉收益。**这正是要测的东西**：小模型 + 慢互联，TP=2 很可能比 TP=1 更慢。报告里必须和"两个独立单卡实例"区分开，不能混称。

**E7b 双实例（吞吐导向，可选）**

两个进程分别绑 GPU 0 和 GPU 1，各跑一半请求，比较总吞吐。与 TP 不是一个实验，结果不能混排。

### 5.3 Server 模式（Kaggle）

```bash
!CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct --served-model-name qwen1.5b \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 1024 --max-num-seqs 16 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching > /kaggle/working/vllm-server-1.5b.log 2>&1 &
```

等日志出现 `Application startup complete` 再压测：

```bash
!sleep 90 && grep -c "Application startup complete" /kaggle/working/vllm-server-1.5b.log

!python scripts/benchmark_http.py \
  --model qwen1.5b \
  --prompts experiments/concurrency/prompts_512x64.txt \
  --requests 64 --concurrency 16 --max-tokens 64 \
  --output results/raw/kaggle-1.5b-http-concurrency-16.jsonl
```

Notebook 中断后进程可能还在，重启实验前先查端口和 PID：

```bash
!nvidia-smi --query-compute-apps=pid,used_memory --format=csv
!pkill -f vllm.entrypoints.openai.api_server
```

---

## 6. Kaggle 7B（AWQ / GPTQ）

### 6.1 为什么必须先量化

7B FP16 权重约 15.2GB，单张 T4 可用显存约 15GB，**放不下**（还要 KV cache 和激活）。所以：

- 单卡：必须用 AWQ / GPTQ（约 4.5GB 权重）。
- 双卡 TP=2：FP16 可行（每卡约 7.6GB），作为一条可选对照臂。
- **7B 不做 HF vs vLLM 对比**：`run_hf.py` 不支持量化，单卡 FP16 又放不下。报告里直接写明"7B 的 HF 基线在 T4 单卡上不可行"。

### 6.2 先验证 checkpoint 能加载

不要一上来就跑完整实验，先用最小命令验证：

```bash
!CUDA_VISIBLE_DEVICES=0 python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-7B-Instruct-AWQ', quantization='awq',
          max_model_len=2048, gpu_memory_utilization=0.9)
print(llm.generate(['hello'], SamplingParams(max_tokens=16, temperature=0))[0].outputs[0].text)
"
```

加载失败（找不到仓库名、kernel 不支持）就换 checkpoint 重试，或按失败实验保留日志。**先记下确实能加载的那个仓库名**，后面所有命令都用它。

同样验证 GPTQ：

```bash
!CUDA_VISIBLE_DEVICES=0 python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', quantization='gptq',
          max_model_len=2048, gpu_memory_utilization=0.9)
print(llm.generate(['hello'], SamplingParams(max_tokens=16, temperature=0))[0].outputs[0].text)
"
```

### 6.3 实验

**E6 量化对比（核心）**

```bash
!for Q in "AWQ awq" "GPTQ-Int4 gptq"; do \
  set -- $Q; \
  CUDA_VISIBLE_DEVICES=0 python scripts/run_vllm.py \
    --model Qwen/Qwen2.5-7B-Instruct-$1 --quantization $2 \
    --requests 16 --max-num-seqs 16 --max-model-len 1024 --max-tokens 64 \
    --gpu-memory-utilization 0.9 \
    --output results/raw/kaggle-7b-$1.jsonl --include-text; \
done
```

两个 checkpoint 用同一份 prompt、同样的参数，输出文本互相 sanity check，并和 1.5B FP16 的文本风格对比。

**E2 并发**

```bash
!for SEQ in 16 64; do \
  CUDA_VISIBLE_DEVICES=0 python scripts/run_vllm.py \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq \
    --prompts experiments/concurrency/prompts_512x64.txt \
    --requests 64 --max-num-seqs $SEQ --max-model-len 1024 --max-tokens 64 \
    --gpu-memory-utilization 0.9 \
    --output results/raw/kaggle-7b-awq-concurrency-$SEQ.jsonl; \
done
```

**E3 上下文长度**

7B 的 KV cache 约 56 KB/token，是 0.5B 的 4.7 倍。`gpu_memory_utilization=0.9`（14.4GB）减去 4.5GB 权重，KV 池约 9GB ≈ 16 万 token，所以 16384 单条（917MB）放得下，但 T4 上 7B 的 16K prefill 会非常慢。**建议 7B 的上限做到 8192，16384 标为可选**。

```bash
!for CFG in "512 16 1024" "2048 4 2560" "4096 2 4608" "8192 1 8704"; do \
  set -- $CFG; \
  CUDA_VISIBLE_DEVICES=0 python scripts/run_vllm.py \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq \
    --prompts experiments/context_length/prompts_$1.txt \
    --requests $2 --max-num-seqs $2 --max-model-len $3 --max-tokens 64 \
    --gpu-memory-utilization 0.9 \
    --output results/raw/kaggle-7b-awq-context-$1.jsonl; \
done
```

**E4 / E5**

把上面命令的 `--model/--quantization` 换掉、`--prompts` 换成 `prefix_caching/` 或 `chunked_prefill/` 的对应文件即可，参数与 1.5B 一致，只把 `--gpu-memory-utilization` 改成 0.9。

**E7 TP=1 vs TP=2**

```bash
# AWQ + TP=2
!CUDA_VISIBLE_DEVICES=0,1 python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq \
  --tensor-parallel-size 2 \
  --prompts experiments/concurrency/prompts_512x64.txt \
  --requests 64 --max-num-seqs 64 --max-model-len 1024 --max-tokens 64 \
  --gpu-memory-utilization 0.9 \
  --output results/raw/kaggle-7b-awq-tp2.jsonl

# 可选：FP16 + TP=2（每卡约 7.6GB 权重，显存紧张）
!CUDA_VISIBLE_DEVICES=0,1 python scripts/run_vllm.py \
  --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 \
  --requests 8 --max-num-seqs 8 --max-model-len 2048 --max-tokens 64 \
  --gpu-memory-utilization 0.9 \
  --output results/raw/kaggle-7b-fp16-tp2.jsonl
```

FP16+TP2 这一臂有 OOM 风险，**失败也要保留**（项目规则）。

---

## 7. 结果汇总与报告

### 7.1 汇总

```bash
cd ~/vllm-benchmark-lab

# 本机
$PY scripts/summarize_results.py results/raw/local-0.5b-*.jsonl \
  --gpu results/raw/gpu-local-0.5b-http-16.jsonl \
  --output-md results/processed/summary-local-0.5b.md \
  --output-csv results/processed/summary-local-0.5b.csv

# Kaggle（在 notebook 里跑，或者把结果拉回本机再跑）
$PY scripts/summarize_results.py results/raw/kaggle-1.5b-*.jsonl \
  --output-md results/processed/summary-kaggle-1.5b.md
```

输出两张表：

- **Runs 表**：每行一个 run，含 `latency_mode` 列。`batch` 表示这是离线 batch 数字（`latency_ms` 是 batch 延迟、`out_tok_per_s` 是 batch 吞吐、百分位不可用）；`per-request` 才看 p95/TTFT/TPOT。
- **GPU 表**（带 `--gpu`）：每卡峰值显存、峰值显存占比、平均/最大利用率、最大功耗、平均温度。

`latency_mode=batch` 的 run 会被自动隐藏百分位。这一步就是为了防止把 batch 延迟当成 p95 写进报告。

### 7.2 填报告

`docs/benchmark_report.md` 的对应关系：

| 报告小节 | 数据来源 |
| --- | --- |
| 2. Environment | `results/raw/env-kaggle-*.json` + 每次 run 的 metadata 行 |
| 3. Workload | prompt 文件的 `*.manifest.json` |
| 5. HF vs vLLM | E1a（单请求对单请求）+ E1b（批量收益） |
| 6. Concurrency | E2 + 7.1 的 GPU 表（显存上限/OOM 点）|
| 7. Context Length | E3，注明**实际 input_tokens** 而不是 `max_model_len` |
| 8. Prefix Caching | E4 的 shared vs random 两臂 |
| 9. Chunked Prefill | E5，必须标注 V0 引擎 |
| 10. Quantization | E6，把"不支持/资源不足/实验结果"三类分开写 |

必须写进报告的限制：

1. 本机 4060 与 Kaggle T4 结果**不可直接横向比较**（不同 GPU、不同 dtype：4060 是 bfloat16，T4 是 float16）。
2. 离线数字是 batch 级；per-request 数字来自 server 模式，两者不能混在同一张表里比较。
3. TP=2、两个单卡实例、单卡运行是三种不同实验。
4. chunked prefill 的结论限于 V0；V1 下该特性恒开。

---

## 8. 检查清单

跑之前：

- [ ] `$PY scripts/make_prompts.py --check <文件>` 长度符合预期
- [ ] prompt 文件条数 == `--requests`（坑 3）
- [ ] `--max-model-len` ≥ prompt 实际长度 + `--max-tokens`（坑 4）
- [ ] prefix caching 的臂确认了开关状态（坑 2）
- [ ] chunked prefill 的臂用了 `VLLM_USE_V1=0`（坑 1）
- [ ] V0 的臂满足 `max_num_batched_tokens ≥ max_model_len`（坑 6）
- [ ] fp8 KV cache 的臂用了 `VLLM_USE_V1=0`（坑 7）
- [ ] 每条命令的引擎版本（V0/V1）已记进报告——**JSONL 里没有这个字段**（坑 7）
- [ ] Kaggle 上 `VLLM_USE_V1` / `HF_HOME` 在任何 import 之前设置

每次 run 之后：

- [ ] 记下退出码（`echo $?`），OOM 也要保留
- [ ] GPU 采集器覆盖了加载、warmup、推理三个阶段
- [ ] `results/raw/` 文件按命名规范落盘

时间预算（粗估）：

| 阶段 | 本机 4060 | Kaggle（每 session）|
| --- | --- | --- |
| 模型下载 | 已缓存 | 首次 5–10 分钟 |
| server 首次启动 | ~2 分钟 | ~3 分钟 |
| E1–E7 全部（0.5B）| 1–2 小时 | — |
| E1–E7 全部（1.5B）| — | 2–3 小时 |
| E6/E7（7B）| — | 2–4 小时（prefill 很慢）|
| 汇总与写报告 | 1 小时 | 1 小时 |

Kaggle 配额：单次 session 上限 12 小时，GPU 每周 30 小时。建议**一个模型一个 notebook**，避免重复下载权重；结果目录存成 Dataset 便于下次复用。

---

## 9. 本次新增/修改的脚本

| 文件 | 说明 |
| --- | --- |
| `scripts/make_prompts.py` | **新增**。生成精确 token 长度的 prompt（distinct / shared-prefix / random-prefix / mixed），带 `--check` 复核和 `--strict` 保证 |
| `scripts/benchmark_http.py` | **新增**。走 OpenAI 兼容 server 压测，输出 per-request TTFT / TPOT / p50 / p95 / p99 |
| `scripts/summarize_results.py` | **新增**。把 `results/raw/*.jsonl` 聚合成报告用 markdown/CSV，自动标注 batch / per-request 语义 |
| `scripts/run_vllm.py` | 新增 `--max-num-batched-tokens`（chunked prefill A/B 必需）；注释说明 prefix caching 显式传值的原因 |
| `scripts/common.py` | 新增 `percentile()`，供新脚本复用 |
| `scripts/benchmark.py` | 改用 `common.percentile`，删掉重复实现 |

---

## 10. 本次实验（0.5B 本机全矩阵）暴露的待办

按优先级排序，都是上面的坑直接推出来的：

1. **把 `VLLM_USE_V1` 写进 JSONL**（坑 7）。`environment_metadata()` 目前不记这个变量，
   导致事后无法判断任一 run 跑在 V0 还是 V1 上——而坑 1、6、7 全都依赖它。
   这是**最该先补的一条**，否则已经跑完的 AWQ 臂永远无法归因。
2. **量化一节补一条同引擎、同 `gpu_memory_utilization` 的 FP16 基线**（坑 7）。
   现在 AWQ / fp8 KV 只能和 `vllm-batch16`（V1、0.90）比，差 13% 却归因不了。
3. **E5 的 ON 臂补 server 模式测量**。离线 batch 只能给出整体耗时，
   看不出"短请求是否被长 prefill 拖慢"——而这正是 chunked prefill 的卖点。
4. **7B / 1.5B 的 Kaggle 部分尚未开始**，本机只有 0.5B。报告第 2 节的环境表要留出 Kaggle 行。
5. **`docs/kaggle_runbook.md` 是早期版本**，与 `runbook_three_models.md` 第 5、6 节重复且更旧；
   建议合并或标注为历史文档，避免两处口径不一致。
