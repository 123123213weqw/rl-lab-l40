# 能不能练更大的模型：先修配置，再谈尺寸

日期：2026-10-26。基线：`rl-grpo8b-baseline-20261016` / Qwen3-8B 全参 GRPO / 4×L40。

## 结论

**能，但当前瓶颈不是显存容量，是 offload 配置。** 同样的 4 张卡，
关掉 `param_offload` / `optimizer_offload` 后：

| | 原配置 | 关 offload | 提升 |
|---|---|---|---|
| 每步耗时 | 920 s | **247 s** | **3.72×** |
| update_actor | 605 s | 146 s | 4.14× |
| ref | 147 s | 34.7 s | 4.24× |
| old_log_prob | 131 s | 33.1 s | 3.96× |
| gen | 24 s | 22.2 s | 1.08× |
| 新增 update_weights | — | 10.5 s | — |
| MFU | 0.40% | **1.88%** | 4.7× |
| GPU 利用率 | ~0% | **99–100%** | — |
| 显存 reserved | — | 36.1 GB/卡 | — |
| CPU 内存 | — | 97 GB | — |

50 步：**12.8 小时 → 3.4 小时**。等于同样的卡多出 3.7 倍的实验吞吐。

## 原配置为什么慢

`launch.json` 里：

```
actor.fsdp_config.param_offload=true
actor.fsdp_config.optimizer_offload=true
actor.fsdp_config.model_dtype="float32"
actor.ppo_micro_batch_size_per_gpu=1        ← 每卡 micro batch 只有 1 条
ref.log_prob_micro_batch_size_per_gpu=1
ref.fsdp_config.param_offload=true
```

参数被 offload 到 CPU，显存里只剩当前分片，因此 micro batch 只能压到 1，
GPU 全程在等 PCIe 搬运。拓扑上这台机器**没有 NVLink**（GPU0-3 与 GPU4-7
是两组 PCIe switch，跨组还走 NUMA 的 UP)，offload 的代价被进一步放大。

对照实验中其它设置全部不变，只改这四项：
`param_offload=false`、`optimizer_offload=false`、
`ppo_micro_batch_size_per_gpu=4`、`model_dtype="bfloat16"`。

## 各尺寸的显存账

按 FSDP 全分片、bf16 参数/梯度、fp32 Adam(m+v)、不含激活估算：

每卡显存 ≈ P × (2+2+8)/N + ref P×2/N + vLLM P×2/TP_per_card

| 模型 | 参数 | 4 卡 | 8 卡 | 判定 |
|---|---|---|---|---|
| Qwen3-8B | 8.2B | 24 GB | 12 GB | 4 卡轻松 |
| Qwen3-14B | 14.8B | **44.4 GB**，贴 46 GB 上限 | **22.2 GB** + ref 3.7 + vLLM(TP=4) 7.4 ≈ 33 GB | **8 卡可行** |
| Qwen3-32B | 32.8B | 不行 | **49.2 GB > 46 GB** | 必须 offload Adam 或 LoRA |
| Qwen3-30B-A3B | 30.5B | 不行 | 45.8 GB，贴边 | 必须 offload |

要点：
- **不 offload 的全参上限约 14B**（8 卡）。这是这台机器做全参 GRPO 的现实天花板。
- 32B 要跑，只能：Adam offload（32.8 GB 搬到 CPU，速度会退回接近原配置）、
  或 LoRA（基座只存 bf16 不存梯度/Adam，8 卡约 25 GB/卡，可行且快）。
- **MoE 在这里没有优势**。30B-A3B 虽然激活只有 3.3B，算得快，
  但 offload 搬运的是全量参数，量级与 32B 密集模型相同。
  只有配 expert parallel 才能拿到稀疏红利。

## 磁盘约束

`/data1` 已用 91%，只剩 355 GB。权重放这里要小心：
- 14B bf16 ≈ 30 GB
- 32B bf16 ≈ 65 GB
建议权重放 `/data`（2.5 TB 可用）或 `/myjfs`（6 TB），checkpoint 也一并放。

## 但先回答一个更前置的问题

上一轮已经测出：这个 8B GRPO baseline 的 **+19.64pp 全部来自格式合规**，
在免格式口径下是 +0.30pp（p=0.689），ARC-Challenge 上零变化。

**换更大模型不会修复这个问题，反而可能让现象消失得更彻底**：
Qwen3-14B / 32B 这类更"听话"的 base，本来就更容易守 `#### N` 格式，
于是 GRPO 的可见增益会更接近 0。

所以真正的排序是：

1. **修配置**（已在本次验证，3.72×，零成本）
2. **修奖励设计**——现在是「格式违规直接 0 分」，格式成了最大梯度，
   3 步就饱和。真正的推理提升需要一个不把格式当主信号的奖励。
3. **再考虑换大模型**——只有当前两步做完，更大模型才有意义。

## 复现

```bash
cd $RL_LAB
# 4 卡、无 offload、micro_batch 4、跑 3 步
NCPU=40 MICRO=4 RMICRO=4 bash run_probe.sh noffload4 4 3
```

`run_probe.sh` 复刻了 `train-001` 的全部 99 个参数，只把上述四项暴露成环境变量
（`POFF` / `OOFF` / `MICRO` / `RMICRO` / `MDTYPE` / `TP`），便于逐项对照。

## 踩过的坑

- ray 默认 `num_cpus=16` 不够，8 卡时报
  `No available node types can fulfill resource request {'CPU': 24.0, 'GPU': 8.0}`
- `reward.custom_reward_function.path` + `.name` 必须显式传，否则
  `NotImplementedError: Reward function is not implemented for data_source='gsm8k_baseline'`
- `pkill -f "verl.trainer.main_ppo"` 会匹配到自己那条 ssh 命令行（把连接杀掉），
  用 PID 或 `main[_]ppo` 规避
