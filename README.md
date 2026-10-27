# rl-lab

在 8 × L40 46GB 上做 LLM 强化学习后训练的实验记录。

起点是审计一个已有的 Qwen3-8B GRPO baseline，终点是跑通代码任务的 GRPO。
每份结论都附带可复现命令和原始数据；被推翻的结论保留下来，不删。

## 结论摘要

按发现顺序，每条都推翻了上一条的前提：

| # | 问题 | 结论 |
|---|---|---|
| 1 | 那个 8B GRPO 到底提升了什么？ | **什么都没提升。** +19.64pp 全部来自格式合规；免格式口径 +0.30pp（p=0.689），ARC 上零变化 |
| 2 | 训练慢吗？ | **慢 3.7 倍，但不是显存问题。** 关掉 offload 后 920→247 s/step，MFU 0.40%→1.88% |
| 3 | 为什么学不动？ | **84% 的组零方差。** GSM8K 上 86% 的题 8 次全对，G=4 时 61.7% 的算力在算零梯度 |
| 4 | 换更难的评测？ | **不行。** 越难题越少，AIME(n=30) 的 MDE 是 22.9pp，测不出 5pp |
| 5 | 换代码数据？ | **对，但仍需过滤。** TACO pass@1 只有 25.3%，60.5% 的题全错——太难 |
| 6 | dense 奖励有用吗？ | **有用但有限。** 零方差组 22.0%→14.0%，不是决定性的 |
| 7 | 8 卡能跑吗？ | **显存够（28.7/46 GB），但初始化不稳定。** 跑通过一步，现在复现不出来 |
| 8 | 训练跑通了吗？ | **跑通了，但没学到东西。** step12 vs base +0.91pp，95% CI 含 0 |
| 9 | 为什么第一次训练越训越差？ | **漏了 `enable_thinking=False`。** 96.5% 的预算耗在思考上，代码写不完 |

## 目录结构

```
paths.py           统一路径解析（所有脚本从这拿 DATA/RESULTS）
sandbox/           沙箱：执行模型生成的不可信代码
dataset/           TACO 数据流水线：验证 → 构建 → 判分 → 切分
train/             GRPO 训练：奖励函数 + 启动器
eval/              离线评测 + 配对检验 + 单测
analyze/           诊断脚本：可学空间、难度、奖励密度、thinking 对比
docs/              10 份结论文档
data/              数据集（大文件 gitignore，可由脚本重建）
results/           评测与统计输出
```

`taco_lib.py` 通过 `Path(__file__).parent` 定位 `driver.py` 和 `sandbox_run.sh`，
三者必须同目录 —— 这是 `sandbox/` 单列的原因。

## 核心组件

### 沙箱 `sandbox/sandbox_run.sh`

没有 docker 权限、没有 nsjail/bwrap，用内核特性自建：

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

### 数据流水线 `dataset/`

```
validate_gold.py      验证 TACO 测试用例可信（gold 解必须全过）
  → build_code_dataset.py  剔除坏源、去重、哈希切分
  → grade_candidates.py    基座采样 + 沙箱判分，得 pass rate
  → build_splits.py        按 pass rate 过滤出训练集与评测基准
  → prepare_code_data.py   转成 veRL parquet
```

### 训练 `train/`

`code_reward.py` 按**通过测试数 / 总测试数**给分，而不是 0/1 门控。
这是代码任务相对数学任务的核心优势：通过 15/20 个测试是可观测的中间信号。

`launch_code_grpo.py` 生成 85 个 Hydra 参数并启动 veRL。路径读环境变量
`GRPO_BASELINE`，不硬编码。

### 评测 `eval/`

`eval_checkpoint.py` 在 326 题基准上评测，保存逐题数据；
`paired_test.py` 做 10,000 次配对 bootstrap。用配对而非总体均值比较，
能消掉题目难度方差——同样的题量下灵敏得多。

指标用**多次采样的平均 pass rate**而非二元 pass@1：

| 口径 | MDE（326 题） |
|---|---|
| 二元 pass@1 | 6.9pp |
| pass rate（8 采样） | **2.4pp** |

等价于把题量放大 8 倍。对比 MATH-500 的 5.6pp、AIME 的 22.9pp。

## 关键数字

**TACO 测试数据可信率 65.0%**（9,419 / 14,498），来源分化极大：

| 来源 | 可信率 |
|---|---|
| hackerrank | 93.0% |
| atcoder | 78.6% |
| codeforces | 78.0% |
| aizu | 68.9% |
| codechef | 56.2% |
| hackerearth | 22.2% |
| geeksforgeeks | 0.1% |

`geeksforgeeks` 980 题只过 1 题——整源剔除，不逐条挑。

**数据集**：训练 1,333 题（G=8 零方差 3.0%），评测 326 题（零方差 8.5%）。

**4 卡训练速度**：约 530 s/step（`max_response_length=4096`，64 样本/步）。
12 步 ≈ 80 分钟。

## 复现

所有命令在仓库根执行。

```bash
export HF_ENDPOINT=https://hf-mirror.com
export GRPO_BASELINE=/path/to/grpo-baseline     # 含 vendor/verl 与 execution/*/model

# 1. 沙箱与奖励的单元测试（不需要 GPU）
PATH=$PWD/.tools/bin:$PATH python eval/test_code_reward.py

# 2. 数据流水线
python dataset/validate_gold.py --max-try 5 --workers 48 --out results/taco_validated.json
python dataset/build_code_dataset.py --out data/taco_pool.jsonl
#   → 测 pass rate（需要 GPU，8 卡分片见 train/run_sharded.sh）
python dataset/build_splits.py
python dataset/prepare_code_data.py

# 3. 训练（会先清残留 Ray）
./train/cleanray.sh
python train/launch_code_grpo.py --gpus 0,1,2,3 --steps 12 \
    --max-tokens 4096 --batch-size 8 --group-size 8 \
    --out runs/code-nothink

# 4. 合并 FSDP 分片 + 评测
V=$GRPO_BASELINE/vendor/verl
PYTHONPATH=$V $V/.venv/bin/python -m verl.model_merger merge --backend fsdp \
    --local_dir runs/code-nothink/checkpoints/global_step_12/actor \
    --target_dir merged/step12
python eval/eval_checkpoint.py --model merged/step12 --n 8 --temperature 1.0 \
    --out results/ev12.json
python eval/paired_test.py results/evb.json results/ev12.json
```

## 环境坑（都踩过）

1. **`enable_thinking=False` 必须显式传**。veRL 默认 `apply_chat_template_kwargs: {}`，
   Qwen3 的 chat template 就会开启 thinking，思考过程吃光 response 预算。
   这一项漏掉，训练会「成功」但奖励恒为 0（见 `docs/08`）。
2. **`CUDA_VISIBLE_DEVICES=""` 必须显式设**，即使写 `device="cpu"`。
   torch 2.14 的 `Adam.step()` 会调 `torch.accelerator.current_stream()`，
   触发 CUDA 上下文初始化；显存被占满时直接 OOM。
3. **`VLLM_USE_FLASHINFER_SAMPLER=0`**。`/usr/local/cuda` 的 nvcc 是 11.8，
   flashinfer JIT 需要 `--compress-mode=size`（12.8+），编译必失败。
4. **vLLM JIT 需要 `ninja`**，系统没有；装在 `.tools/bin` 并加 PATH，
   不动既有 vllm 环境。
5. **`sys.set_int_max_str_digits(0)`**。TACO 有 >4300 位的整数字面量，
   触发 Python 的 DoS 防护，`json.loads` 抛 ValueError。
6. **结果用 `__RESULT__` 哨兵前缀**。候选代码会往 stdout 泄漏内容，
   否则被当成结果解析而崩溃。
7. **`pkill -f "ray::"` 会杀掉自己那条 ssh 命令**，用 `ray[:][:]` 规避。
   残留 Ray 集群会导致 8 卡初始化挂住。
8. **`+ray_kwargs.ray_init.num_gpus` 必须显式给**，否则 Ray 认为没有 GPU，
   把所有 actor 塞到 GPU 0。
9. **`ppo_mini_batch_size` 的单位是 prompt 数**，不是样本数。
10. **过滤 pass rate 不能用 Laplace 平滑值**：n=4 时 `(k+1)/(n+2)` 只产生
    五个值且全在 [0.1,0.9] 内，过滤器会静默失效。要用原始比例 `k/n`。

## 当前状态

已完成：数据流水线、dense 奖励、4 卡训练跑通、配对评测框架。

未完成（按重要性）：

- [ ] **跑满一个 epoch**（160 步）。当前 12 步只覆盖 96 个 prompt，占 7.5%
- [ ] **binary vs dense 对照**。dense 的方差优势测过，但没在真实训练里比最终效果
- [ ] **8 卡初始化挂起**的根因（见 `docs/07`，已排除显存/Ray/cgroup/PCIe/参数）
- [ ] `batch_size` 从 8 提到 16（显存 30/46 GB，有余量）

## 许可

Apache-2.0。评测数据派生自 [BAAI/TACO](https://huggingface.co/datasets/BAAI/TACO)
（同样 Apache-2.0），经重新执行参考解过滤后保留；详见 `NOTICE`。

## 局限

- 训练结论来自单一种子，无多种子复现
- 代码题难度依赖当前基座；模型变强后原可学带会向难端移动
- GSM8K 与 TACO 测试集虽未参与训练，但存在预训练污染可能
- TACO 中多解题被丢弃而非顺序无关评分（后者会让评分不可信）
- `docs/09` 的 +0.91pp 不显著，**不能**当作「GRPO 在代码任务上有效」的证据
