# 第一次真实训练：4 卡 12 步，能不能学到东西

日期：2026-10-26/27。运行目录 `runs/code-nothink`。

## 结论先说

**训练跑通了，12 步后比 base 高 0.91pp，但统计上不显著。**
不能声称这个 GRPO 配置在代码任务上学到了东西。

## 修好 thinking 之后的训练过程

12 步全部完成，每步约 390-400 秒，总计约 80 分钟。
截断率从开 thinking 时的 0.83 降到 **0.0156**，全程稳定。

    step  score     clip
    1     0.5258    0.0156
    2     0.5695    0.0156
    3     0.5227    0.0156
    4     0.4359    0.0156
    5     0.5930    0.0156
    6     0.5477    0.0156
    7     0.4578    0.0156
    8     0.3320    0.0156
    9     0.5297    0.0156
    10    0.4606    0.0156
    11    0.5602    0.0156
    12    0.5164    0.0156

均值 0.5043，前 6 步 0.5324，后 6 步 0.4761，
线性斜率 **-0.0036/step** —— 训练集上的 score 没有上升，略微下降。

（训练 score 是在 train split 上算的，不能直接和 eval 集的数字比。）

## 离线评测：326 题 × 8 采样，配对 bootstrap

用 `eval_checkpoint.py` 在 eval split 上评测，10,000 次配对 bootstrap：

| 对比 | 配对差 | 95% CI | 单侧 p | 变好/变坏/不变 |
|---|---|---|---|---|
| **step12 vs base** | **+0.91pp** | [-0.27, +2.09] | 0.067 | 121 / 91 / 114 |
| step6 vs base | +0.64pp | [-0.60, +1.87] | 0.155 | 108 / 91 / 127 |
| step6 vs step12 | +0.27pp | [-0.84, +1.37] | 0.306 | 95 / 95 / 136 |

绝对水平（dense 口径，n=8, t=1.0, thinking off）：

    base    0.4845
    step6   0.4909
    step12  0.4936

三个对比的 95% CI **全部包含 0**。step12 vs base 的 p=0.067 看着接近，
但 CI 含 0，不能声称显著。

step6 -> step12 的 +0.27pp（p=0.306）说明后 6 步基本没动，和训练集上
score 走平一致。

## 这个实验证明了什么、没证明什么

**证实的：**

1. 4 卡全参 GRPO 在 L40 上完全可行，12 步 80 分钟，无崩溃、无 OOM。
2. dense 奖励（按通过测试比例给分）在真实训练里工作正常：
   `score/mean` 在 0.33-0.59 之间，`score/max` 恒为 1.0，`score/min` 为 0。
   部分正确被计分了，这正是二元口径做不到的。
3. `enable_thinking=False` 是必须的。开 thinking 时截断 83%、奖励 0.22；
   关掉后截断 1.5%、奖励 0.50。**同一份数据同一套超参，只差这一行配置。**
4. 沙箱评分在训练循环里稳定：超时率 2.6-2.7%，没有把训练拖死。

**没证实的：**

1. 这个配置能提升代码能力。+0.91pp 且不显著。
2. dense 奖励比 binary 更好。这次只跑了 dense，没做 binary 对照。
3. 步数够了。12 步很可能远不够，训练集 1281 题、batch 8，12 步才碰了
   96 个 prompt（不到 8% 的数据）。

## 为什么很可能只是步数不够

- 训练集 1281 题，`train_batch_size=8`，12 步只覆盖 96 个 prompt。
- 一个 epoch 需要 160 步。当前只跑了 **7.5%**。
- 而 score 的组内方差本来就不小（0.33-0.59 波动主要来自题目难度差异）。

所以「12 步没效果」更可能说明**训练太短**，而不是「GRPO 无效」。
这和 GSM8K 那次不同 —— 那次是奖励设计导致的必然零梯度，这次不是。

## 复现

    cd $RL_LAB
    ./cleanray.sh
    ./.venv-llm/bin/python launch_code_grpo.py \
        --gpus 0,1,2,3 --steps 12 --max-tokens 4096 \
        --batch-size 8 --group-size 8 \
        --out $RL_LAB/runs/code-nothink

    # 合并 + 评测（save_freq=6，会存 step6 和 step12）
    V=$GRPO_BASELINE/vendor/verl
    PYTHONPATH=$V $V/.venv/bin/python -m verl.model_merger merge --backend fsdp \
        --local_dir runs/code-nothink/checkpoints/global_step_12/actor \
        --target_dir $RL_LAB/merged/step12

    PATH=$RL_LAB/.tools/bin:$PATH CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    $VLLM_PY -u eval_checkpoint.py --model merged/step12 \
        --n 8 --temperature 1.0 --out results/ev12_pp.json

    ./.venv-llm/bin/python paired_test.py results/evb_pp.json results/ev12_pp.json

## 下一步该做什么

1. **跑满一个 epoch**（160 步，约 18 小时 @ 4 卡）。当前 12 步只有 7.5%。
2. **做 binary vs dense 对照**。dense 的方差优势已经测过（零方差 22%->14%），
   但没在真实训练里对比最终效果。
3. **提 batch_size**。8 太小，梯度噪声大。显存允许到 16（当前 30/46 GB）。
4. 如果要快，先解决 8 卡挂起（见 `07-eight-gpu.md`），能到 1.6x。
