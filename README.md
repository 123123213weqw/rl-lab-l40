# rl-lab

在 L40 节点（8 × NVIDIA L40 46GB）上做 LLM 强化学习后训练的实验记录。

起点是审计一个已有的 GRPO baseline，终点是构建一套可用的代码数据集。
每份结论都附带可复现的命令和原始数据。

## 结论摘要

按发现顺序，每一条都推翻了上一条的前提：

| # | 问题 | 结论 |
|---|---|---|
| 1 | 那个 8B GRPO 到底提升了什么？ | **什么都没提升。** +19.64pp 全部来自格式合规，免格式口径 +0.30pp（p=0.689），ARC 上零变化 |
| 2 | 训练慢吗？ | **慢 3.7 倍，但不是显存问题。** 关掉 offload 后 920s → 247s/step，MFU 0.40% → 1.88% |
| 3 | 为什么学不动？ | **84% 的组零方差。** GSM8K 上 86% 的题 8 次全对，G=4 时 61.7% 算力在算零梯度 |
| 4 | 换更难的评测？ | **不行。** 越难题越少，AIME(n=30) 的 MDE 是 22.9pp，测不出 5pp |
| 5 | 换代码数据？ | **对，但仍需过滤。** TACO 上 pass@1 只有 25.3%，60.5% 的题全错——太难 |

## 目录

```
docs/
  00-env.md              环境与硬件
  01-grpo-capability.md  审计：GRPO 增益的分解
  02-scaling.md          速度与显存：配置瓶颈 vs 模型尺寸
  03-headroom.md         GSM8K 可学空间
  04-difficulty.md       难度分层与评测功效
  05-code-dataset.md     TACO 数据集构建
```

代码平铺在根目录——`taco_lib.py` 通过 `Path(__file__).parent` 定位
`driver.py` 和 `sandbox_run.sh`，三者必须同级。

## 核心组件

### 沙箱（`sandbox_run.sh`）

执行模型生成的不可信代码。没有 docker 权限、没有 nsjail/bwrap，用内核特性自建：

```
unshare -Urnm --fork --pid --mount-proc
  user / net / mount / pid 四个命名空间
  tmpfs 遮蔽 /home /data1 /data /myjfs /root /opt
prlimit: cpu / as / nproc / nofile / fsize / core
```

对抗性验证（全部通过）：

| 攻击 | 结果 |
|---|---|
| 写 `/data1/…/PWNED` | FileNotFoundError，宿主无文件 |
| 连 `1.1.1.1:53` | OSError |
| fork 炸弹 | 4–12 个后 BlockingIOError |
| 死循环 | 4.03s 被 SIGKILL |
| 分配 4GB | 被 RLIMIT_AS 杀 |

### 数据流水线

```
validate_gold.py       验证 TACO 测试用例是否可信（gold 解必须全过）
  → build_code_dataset.py   剔除坏源、去重、哈希切分
  → grade_candidates.py     基座模型采样 + 沙箱判分，得 pass rate
  → build_splits.py         按 pass rate 过滤出训练集与评测基准
```

## 关键数字

**TACO 测试数据可信率 65.0%**（9,419 / 14,498）。来源分化极大：

| 来源 | 可信率 |
|---|---|
| hackerrank | 93.0% |
| atcoder | 78.6% |
| codeforces | 78.0% |
| aizu | 68.9% |
| codechef | 56.2% |
| hackerearth | 22.2% |
| geeksforgeeks | 0.1% |

`geeksforgeeks` 980 题只过 1 题——整源剔除，不逐题挑。

**评测基准功效**：328 题，平均 pass rate 0.421，G=8 零方差率 8.5%。

| 指标口径 | MDE |
|---|---|
| 二元 pass@1 | 6.9pp |
| pass rate（8 采样） | **2.4pp** |

用多次采样的 pass rate 当指标，等价于把题量放大 8 倍。
对比 MATH-500 的 5.6pp、AIME 的 22.9pp。

## 复现

```bash
export HF_ENDPOINT=https://hf-mirror.com

# 1. 验证测试可信（约 5 分钟，48 并发）
./.venv-llm/bin/python validate_gold.py --max-try 5 --workers 48 \
    --out results/taco_validated.json

# 2. 构建池子（剔除坏源 + 去重 + 切分）
./.venv-llm/bin/python build_code_dataset.py --out data/taco_pool.jsonl

# 3. 测 pass rate（需要 GPU；8 卡分片见 run_sharded.sh）
PATH=$PWD/.tools/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES=0 \
$VLLM_PY grade_candidates.py \
    --validated /tmp/trainset.json --model <基座ckpt> --n 4 --graders 40 \
    --out results/train_passrate.json

# 4. 过滤出训练集与评测基准
./.venv-llm/bin/python build_splits.py
./.venv-llm/bin/python power_analysis.py
```

## 环境坑（都踩过）

1. **`CUDA_VISIBLE_DEVICES=""` 必须显式设**，即使写 `device="cpu"`。
   torch 2.14 的 `Adam.step()` 会调 `torch.accelerator.current_stream()`，
   触发 CUDA 上下文初始化；显存被占满时直接 OOM 崩溃。
2. **`VLLM_USE_FLASHINFER_SAMPLER=0`**。机器上 `/usr/local/cuda` 的 nvcc 是 11.8，
   flashinfer JIT 需要 `--compress-mode=size`（12.8+），编译必失败。
3. **vLLM JIT 需要 `ninja`**，系统没有；装在 `.tools/bin` 并加 PATH，
   不改动既有的 vllm 环境。
4. **`sys.set_int_max_str_digits(0)`**。TACO 有 >4300 位的整数字面量，
   触发 Python 的 DoS 防护，`json.loads` 抛 ValueError。
5. **结果用 `__RESULT__` 哨兵前缀**。候选代码会往 stdout 泄漏内容，
   否则会被当成结果解析而崩溃。
6. **`pkill -f "verl.trainer.main_ppo"` 会杀掉自己那条 ssh 命令**，
   用 `main[_]ppo` 或 PID 规避。
7. **Ray 需要足够 `num_cpus`**，默认 16 不够 8 卡启动。

## 待办

- [ ] 完成 train split 的 pass rate 测量（进行中，4 卡分片）
- [ ] 过滤后训练集落盘为 `data/train_code.jsonl`
- [ ] 奖励改为「通过测试数 / 总测试数」而非二元对错
- [ ] 格式 SFT 对照臂：证明原 GRPO 的边际贡献为零

## 许可

Apache-2.0。评测数据派生自 [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO)（同样
Apache-2.0），经重新执行参考解过滤后保留；详见 `NOTICE`。

## 局限

- 结论均来自单一训练种子，无多种子复现
- 代码题难度依赖当前基座模型；模型变强后原可学带会向难端移动
- GSM8K 与 TACO 测试集虽未参与训练，但存在预训练污染可能
- TACO 中多解题被丢弃而非被正确评分（顺序无关比较会让评分不可信）
