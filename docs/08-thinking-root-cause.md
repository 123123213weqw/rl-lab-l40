# 截断的真凶：漏了 enable_thinking=False

日期：2026-10-26。这份文档记录一次 12 步训练的失败与定位。

## 现象

第一次 4 卡 12 步训练（`runs/code-4gpu`）跑完了，1h48m，无崩溃：

    step  score     clip
    1     0.1391    0.83
    2     0.4062    0.59
    3     0.2812    0.70
    4     0.3750    0.62
    5     0.2656    0.73
    6     0.3883    0.59
    7     0.1875    0.81
    8     0.0000    1.00   <- 全截断
    9     0.0000    1.00   <- 全截断
    10    0.2500    0.73
    11    0.0938    0.91
    12    0.2344    0.77

score 均值 0.2184，前 6 步 0.3092，后 6 步 0.1276。
线性斜率 **-0.0170/step**，12 步累计 -0.187。

**没有提升，反而在退化。**

## 诊断：奖励被截断完全主导

score 与截断率的相关系数 **r = -0.9970**。

这不是"部分相关"，是几乎完全决定。step 8/9 截断率 1.00 时 score 恰好 0.00。
平均输出长度 3748 token，上限 4096 —— 一直贴着天花板。

也就是说：模型在答代码题，但**奖励实际测量的是"有没有写完"**，
不是"代码对不对"。这和 GSM8K 那次奖励被格式主导（98.8% 增益来自格式）
是同一个结构性错误，只是主导变量换了。

## 根因：Qwen3 默认开 thinking

baseline 的 `launch.py` 里有这一行：

    "+data.apply_chat_template_kwargs.enable_thinking": c["enable_thinking"]

而 `baseline.json` 里 `enable_thinking: false`。

我的 `launch_code_grpo.py` 里这个配置 **0 处**。而 veRL 的默认是：

    apply_chat_template_kwargs: {}

空字典 -> 不传 enable_thinking -> 用 Qwen3 chat template 的默认值
-> **thinking 打开**。

于是模型每个回复先写几千 token 的思考过程，4096 上限被吃光，
代码根本没开始写。截断 -> 语法不完整 -> 沙箱全判 0。

## 修复与验证

加一行：

    "+data.apply_chat_template_kwargs.enable_thinking": False

重跑（`runs/code-nothink`）：

| | 开 thinking | 关 thinking |
|---|---|---|
| step 1 score | 0.139 | **0.526** |
| step 2 score | 0.406 | **0.570** |
| 截断率 | 0.83 | **0.015** |
| 平均输出长度 | 3748 token | **260 token** |
| score 与截断相关 | -0.997 | 不再成立 |

截断率从 83% 降到 **1.5%**，输出长度从 3748 降到 **260** token。
同样 4096 的上限，之前完全不够，现在绰绰有余。

## 为什么这个错误很难发现

1. 训练**没有报错**。12 步全部"成功"完成，指标齐全。
2. 只看到 `critic/score/mean` 在 0.1-0.4 之间波动，看起来像"训练不稳定"，
   而不像"配置错了"。
3. 真正暴露问题的是两个指标组合看：
   `response_length_non_aborted/mean` 贴着 `max_response_length`，
   且 `clip_ratio` 在 0.6-1.0。**只看 score 永远发现不了。**
4. 我自己的 `measure_reward_density.py` 里设了 enable_thinking=False，
   所以测出来的 pass@1 24.88% 是正常的。训练和测量用了不同配置，
   这个不一致本身就是隐患。

## 教训

- 从 baseline 抄配置要**逐项对照**，漏一项可能就是隐蔽的结构性错误。
  我补齐了 84 个参数，偏偏漏了这一个。
- 日志里必须同时盯 `score` 和 `clip_ratio`。只看 reward 会被骗。
- 训练和离线测量必须用**完全相同的** chat template 配置。

## 待补

- 关 thinking 的 12 步跑完后，趋势是否单调上升（当前只有前 2 步）
- 需要同口径 base 基线：n=8、t=1.0、thinking off、dense reward
- 之前那次 12 步的 checkpoint 没存（save_freq=-1），现在已改为每 6 步存
