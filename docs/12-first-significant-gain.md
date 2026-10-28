# 第一次显著增益：+6.73pp

日期：2026-10-28。运行 `runs/epoch1`（4 卡，167 步为 1 epoch）。
本文件记录 step50 checkpoint 的评测结果。

## 结论

**step50 比 base 高 6.73 个百分点，统计显著。**

| | base | step50 | 差 |
|---|---|---|---|
| **dense**（通过测试比例） | 0.4845 | **0.5518** | **+6.73pp** |
| binary（全过比例） | 0.3854 | 0.4486 | +6.33pp |
| 平均输出长度 | 254.4 token | 261.0 token | +2.6% |
| 超时率 | 2.6% | 2.6% | 0 |

配对 bootstrap（326 题 × 8 采样，10,000 次重采样）：

    配对差 +0.0673  95% CI [+0.0480, +0.0876]  单侧 p < 0.0001
    变好 146 / 变坏 75 / 不变 105

## 和之前那个「不显著」的对比

`docs/09` 记录的 12 步结果：

| | 12 步（batch 8, G 8） | 50 步（batch 8, G 8） |
|---|---|---|
| 配对差 | +0.91pp | **+6.73pp** |
| 95% CI | [-0.27, +2.09] | [+4.80, +8.76] |
| p | 0.067 | < 0.0001 |
| 判定 | 不显著 | **显著** |

**提升 7.4 倍。** 这坐实了 `docs/09` 里的判断：12 步没效果是**步数不够**，
不是 GRPO 无效。当时 12 步只覆盖 96 个 prompt（占数据 7.5%）。

## 关键：这次不是「长度假象」

`docs/01` 审那个原 baseline 时发现，+19.64pp 全部来自格式合规，
一个正则就覆盖了 98.8% 的增益。`docs/10` 又证明 thinking 会让
输出长度涨 20 倍。

所以这次必须排除「模型只是学会了写更长」：

    输出长度 base 254.4 -> step50 261.0  (+2.6%)
    超时率   2.6% -> 2.6%（不变）

长度几乎没动，pass rate 涨了 6.73pp。**增益来自代码正确性，不是长度。**

## 增益是广泛的，不是几个题撑起来的

按难度分层（题干标注的难度）：

| 难度 | n | base | step50 | 差 |
|---|---|---|---|---|
| EASY | 136 | 0.5706 | 0.6526 | **+8.20pp** |
| MEDIUM_HARD | 53 | 0.3680 | 0.4834 | **+11.53pp** |
| MEDIUM | 38 | 0.4482 | 0.4839 | +3.57pp |
| HARD | 32 | 0.3541 | 0.3951 | +4.10pp |
| VERY_HARD | 19 | 0.3026 | 0.4049 | **+10.23pp** |
| UNKNOWN | 48 | 0.5570 | 0.5582 | +0.12pp |

除 UNKNOWN 外每一层都有提升，中高难度提升更大。

**稳健性检验**——逐步剔除增益最大的题，看是否还显著：

| 剔除 | 剩余题数 | 配对差 | 95% CI | 判定 |
|---|---|---|---|---|
| top 0 | 326 | +6.73pp | [+4.80, +8.76] | 显著 |
| top 5 | 321 | +5.67pp | [+3.92, +7.45] | 显著 |
| top 10 | 316 | +4.77pp | [+3.18, +6.39] | 显著 |
| **top 20** | **306** | **+3.45pp** | **[+2.04, +4.90]** | **显著** |

即使砍掉 20 个贡献最大的题（占增益的 52%），仍有 +3.45pp 且显著。
增益不是被少数题撑起来的。

## 可复现性

base 在两次独立评测中完全一致：

    旧 eval dense 0.484516
    新 eval dense 0.484516
    逐题一致 326 / 326

同 seed（311）、同分辨率（n=8, t=1.0, max_tokens=2048）下评测确定性良好。

## 训练侧

step50 训练时的指标（在 train split 上，不能和 eval 数字直接比）：

    step 52  score 0.5742
    step 55  score 0.4273
    step 59  score 0.3234
    均值 0.5351，线性斜率 +0.0004/step

训练集上的 score 走平（斜率接近 0），但 eval 集上显著提升。
两者不矛盾：train split 是过滤后 pass rate 在 [0.1,0.9] 的题，
难度分布和 eval split 不同；而 train score 的逐步波动主要来自
每步抽到的题目难度差异，不是策略退化。

KL 从 0.00054 漂到 0.00189（3.5 倍），仍在 0.001 系数下的合理范围。
截断率恒定 1.56%。

## 局限

1. **只有 step50，不是最终模型。** 训练还在跑（167 步），step50 是中途点。
2. **单种子。** 没有多种子重复，无法区分「GRPO 有效」和「这个种子走运」。
3. **没做 binary vs dense 对照。** 这次只跑了 dense 奖励，
   不知道二元奖励是否也能到同样水平。
4. **评测集是自建的。** 326 题从 TACO 过滤而来，虽与训练集 id 和题干都不相交，
   但同源，可能存在分布相似性。
5. **没有 pass@k。** 还不能判断这是「能力扩张」还是「分布锐化」——
   这正是下一步要回答的。

## 复现

    # 合并 step50
    V=$GRPO_BASELINE/vendor/verl
    PYTHONPATH=$V $V/.venv/bin/python -m verl.model_merger merge --backend fsdp \
        --local_dir /data/wangyue/rl-lab-ckpt/epoch1/global_step_50/actor \
        --target_dir /data/wangyue/rl-lab-ckpt/merged/step50

    # 评测（base 与 step50 同口径）
    PATH=$RL_LAB/.tools/bin:$PATH CUDA_VISIBLE_DEVICES=4 VLLM_USE_FLASHINFER_SAMPLER=0 \
    $VLLM_PY eval/eval_checkpoint.py --model /data/wangyue/rl-lab-ckpt/merged/step50 \
        --n 8 --temperature 1.0 --max_tokens 2048 --out results/ev50.json

    ./.venv-llm/bin/python eval/paired_test.py results/evbase.json results/ev50.json
