# 代码任务的奖励函数

日期：2026-10-26。相关文件：`code_reward.py`、`launch_code_grpo.py`、
`prepare_code_data.py`、`measure_reward_density.py`。

## 动机

前序结论：原 GSM8K 奖励是「格式合规 + 答案正确」的 0/1 门控，在 93% 已饱和的
题目上，组内样本几乎全同分，GRPO 优势项恒为零。GSM8K 上 G=4 时 61.7% 的组
零方差，实测第 10 步那批 84.4%。

代码任务有一个数学任务不具备的性质：一道题通过 20 个测试里的 15 个，
这个「部分正确」本身是可观测的。所以奖励不必是 0/1。

## 实现

`code_reward.compute_score(data_source, solution_str, ground_truth, extra_info)`

    dense  (默认)  score = 通过测试数 / 总测试数
    binary         score = 1.0 if 全过 else 0.0

环境变量控制，便于做对照实验：
`CODE_REWARD_MODE`、`CODE_REWARD_TIMEOUT`、`CODE_REWARD_MAX_TESTS`。

其余设计：

- 从模型输出抽**最后一个**围栏代码块（模型常先给思路再给最终代码）
- 模型代码一律进沙箱执行，reward 模块自己不 import 候选代码
- `data_source != "taco_code"` 直接抛异常，不做静默兜底
- 坏 ground_truth / 空 tests / 无代码，都返回 0 并打标记，不中断训练

`test_code_reward.py` 覆盖 18 个用例全部通过：正常评分、部分通过、全错、
语法错、死循环、无代码、空响应、坏 ground_truth、空 tests、data_source 校验、
binary 模式、多围栏取最后一个。

## 效果实测（已跑通 4 卡）

4 卡实测：score/mean = 0.09375、score/max = 1.0 ——
**dense 奖励确实给出非零部分分**。但 response_length/clip_ratio = 0.906，
90.6% 的代码被 2048 token 上限截断，这是当前奖励偏低的主因。

## 方差实验：比预期保守

300 道题 × 8 采样，同一批样本同时算两种口径：

| | 二元 | dense |
|---|---|---|
| 零方差组 | 66 / 300 = **22.0%** | 42 / 300 = **14.0%** |
| 有效组 | 78.0% | **86.0%** |

二元口径的零方差组里，**36.4%（24 组）在 dense 下重新获得梯度**。

结论：dense 有效，但不是决定性的。22% → 14%，不是「大幅改善」。
剩余 14% 的零方差是真没有可学信号（全对或全错），换口径救不回来。

## 真正的瓶颈是显存，不是奖励

**这个实验在当前硬件上跑不起来。**

全参 FSDP 训练 Qwen3-8B 要约 24 GB/卡（参数 16 + 梯度 16 + fp32 Adam 32，
分片到 4 卡）。vLLM 推理要另加约 19 GB 起（16 GB 权重 + KV cache）。
L40 只有 46 GB。

实测轨迹：

| gpu_memory_utilization | max prompt+resp | 结果 |
|---|---|---|
| 0.40 | 2048 + 1536 | `No available memory for the cache blocks` |
| 0.43 | 2048 + 1024 | `max seq len` 装不下 |
| 0.50 | 1024 + 1024 | 跑到 rollout，GPU0 到 45.2/46 GB，OOM |
| 0.55 | 2048 + 1536 | 直接 CUDA OOM |

`expandable_segments: memory mapping failed with OOM` 反复出现。
即使勉强不崩，45/46 GB 也没有任何余量。

:::warning 本段原结论有误，已修正
原写「8 卡 12 GB 参数 + 12 GB 梯度 + 24 GB Adam ≈ 48 GB，仍超」，
是把总量当成了每卡量。正确的每卡量是 98.3/8 = **12.3 GB**，
加 vLLM 权重共 28.7 GB，在 46 GB 上宽裕。8 卡**确实跑通过一整步**。
详见 docs/07-eight-gpu.md。
:::

真正的问题是：**4 卡显存贴边**（41.0/46 GB），**8 卡初始化不稳定**
（见过成功，但现在复现不出来，挂在 worker group 创建）。

## 接线（veRL）

这个版本走 `reward_loop` 框架，不是旧的 `NaiveRewardManager.__call__`：

    reward.custom_reward_function.path = <repo>/code_reward.py
    reward.custom_reward_function.name = compute_score
    reward.reward_manager.source = "register"
    reward.reward_manager.name   = "naive"
    reward.num_workers = 4

`reward.num_workers` 决定创建多少个 RewardLoopWorker actor，由 `asyncio.gather`
并发；每个 worker 内部按 chunk 串行调 `run_single`。并发度靠这个参数，
不需要自定义 reward manager。

**一开始写的 `ParallelRewardManager`（ThreadPoolExecutor）是多余的** ——
它针对 `verl.workers.reward_manager` 旧接口，而运行时用
`verl.experimental.reward_loop.reward_manager`。已删除。

踩到的坑：`+ray_kwargs.ray_init.num_gpus` 必须显式给，否则 Ray 认为没有 GPU，
把所有 actor 塞到 GPU 0 → OOM。`+ray_kwargs.ray_init.address=local` 和独立
`_temp_dir` 也要给，否则可能去连已有集群而挂住。

## 训练配置里已修的错

| 项 | 错值 | 正解 |
|---|---|---|
| `ppo_mini_batch_size` | `batch_size × group_size`(128) | `batch_size`(8)，单位是 prompt 数 |
| `ray_init.num_gpus` | 未设置 | `= len(gpus)`，带 `+` 前缀 |
| `gpu_memory_utilization` | 0.55 | 0.50（本机器上限，仍不足） |
| reward manager | 自定义 importlib | 内置 `naive` |

## 数据格式

`prepare_code_data.py` → `data/verl/{train,test}.parquet`

    data_source   "taco_code"
    prompt        [{"role":"user","content": INSTRUCTION + "\n\nProblem:\n" + question}]
    ability       "code"
    reward_model  {"style":"rule","ground_truth":"{\"tests\":{\"inputs\":[...],\"outputs\":[...]}}"}
    extra_info    {"id","split","difficulty","source","pass_rate","n_tests","url"}

测试用例走 `reward_model.ground_truth` 而非 `extra_info`：后者在不同 veRL 版本
间传递不稳定，前者是稳定契约。

产出 train 1333 / test 326。`filter_overlong_prompts` 会再滤 1 条
（max_prompt_length=1024 时），实际 1332。

## 下一步

1. 用 LoRA 替代全参，先跑通一个非冒烟实验
2. 若坚持全参，需 8 卡 + Adam offload，并接受速度回退
3. 奖励侧可直接复用，与训练方式无关

## 复现

    cd $RL_LAB
    export HF_ENDPOINT=https://hf-mirror.com

    # 1. 奖励函数单元测试（不需要 GPU）
    ./.venv-llm/bin/python test_code_reward.py

    # 2. 数据准备
    ./.venv-llm/bin/python prepare_code_data.py

    # 3. dense vs binary 的方差对比（需要 GPU）
    PATH=$RL_LAB/.tools/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \\
    CUDA_VISIBLE_DEVICES=0 \\
    $VLLM_PY measure_reward_density.py --model $GRPO_BASELINE/execution/20261017/model \\
        --problems 300 --n 8 --out results/reward_density.json

    # 4. 冒烟（当前会在显存上失败，见上文）
    ./.venv-llm/bin/python launch_code_grpo.py --smoke --gpus 0,1,2,3

## 局限

- dense 奖励的方差收益只在 300 题上测过一次，无重复
- 没有验证 dense 奖励是否引入「针对测试硬编码输出」的 reward hacking；
  测试用例已由 gold 解验证，但模型可以学到部分输出而非通用算法
- 训练从未完整跑通一个 step，所以「dense 让训练更快收敛」这一层没有证据
