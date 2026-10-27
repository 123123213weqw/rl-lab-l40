# analyze

诊断脚本。多数需要 GPU，少数只读已有结果。

| 脚本 | 回答什么问题 |
|---|---|
| `capability_eval.py` | 训练后的模型能力真的更强吗（GSM8K / ARC 对照） |
| `measure_headroom.py` | 题目饱和到什么程度，G 值如何影响有效组比例 |
| `measure_math_difficulty.py` | MATH-500 分难度层后还剩多少可学空间 |
| `measure_reward_density.py` | dense 奖励是否真的产生组内方差 |
| `compare_thinking.py` | thinking 开/关对长度和能力的影响 |
| `power_analysis.py` | 给定题量能测出多小的提升（MDE） |

## 几个反直觉的结论

**GSM8K 已饱和**：250 题 × 8 采样，86% 的题全对。G=4 时 61.7% 的组零方差。
实测训练第 10 步那批更高，达 84.4%。

**换更难的评测走不通**：难度上去题量就下来。AIME n=30、p=23.75% 时
95% CI 宽 30 个百分点，GRPO 的典型增益 3–8pp 完全淹没。

**MATH-500 的 L1 和 GSM8K 是同一个病**：L1 准确率 90.6%、全对率 87.5%，
和 GSM8K 的 93.4% / 86.0% 几乎一样。要 L4/L5 才有可学空间。

**代码任务不是天然落在可学带**：原以为代码通过率会更分散，实测
pass@1 22.2%、67.5% 的题全错——失败方向和 GSM8K 相反（那边是太简单）。
