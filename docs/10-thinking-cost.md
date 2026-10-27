# 关掉 thinking 对 CoT 有没有影响：有，而且方向反直觉

日期：2026-10-27。脚本 `analyze/compare_thinking.py`，原始输出
`results/th2_{on,off}.json`。

## 结论

**关掉 thinking 后推理输出确实消失了，但能力反而更高。**

同一批 80 题、n=4、max_tokens=6144、temperature=1.0：

| | thinking on | thinking off |
|---|---|---|
| **dense（通过测试比例）** | 0.3595 | **0.4916** |
| binary（全过比例） | 0.3563 | 0.3969 |
| 平均输出长度 | 5297 token | **267 token** |
| 中位输出长度 | 6144（贴顶） | 222 |
| 撞到 max_tokens | **63.1%** | 0.3% |
| 含 `</think` 标记 | 38.4% | 0% |

dense 差 **-13.2pp**（开着更差），输出长度差 **19.8 倍**。

## 怎么解读

### 「CoT 没了」是对的

关掉后 0% 的回复含 `</think` 标记，输出从 5297 token 降到 267 token。
模型不再把推理过程写出来。

但要说清：**不是没有推理，是推理不可见**。它仍在一次前向里隐式完成，
只是不落成文本，也没有额外预算。

### 但「没影响」是错的，影响很大

开着 thinking 时 63.1% 的回复撞到 6144 上限。被截断的回复里代码没写完，
沙箱判 0。所以这个对比测的其实是「thinking 在有限预算下的代价」，
不是「thinking 本身的能力上限」。

## 这个对比不完全公平

给 thinking 更多预算（比如 16k）它可能反超。但有两个硬约束：

1. **训练不可行**：平均 5297 token/回复，batch 64 时每步要处理的 token
   是 267 版本的 20 倍。4 卡上每步从 530 s 变成小时级。
2. **截断驱动奖励**：`docs/08` 已经证明，截断率主导奖励（r=-0.997）。
   在预算不足时，模型学到的不是「写对代码」而是「快点收尾」。

所以对 **RL 训练** 来说，关掉 thinking 不是省事，是必需：

| | thinking on | thinking off |
|---|---|---|
| 每步 token 量 | 20x | 1x |
| 截断率 | 63% | 0.3% |
| 奖励信号 | 被截断主导 | 反映代码正确性 |

## 一个可做的后续

如果要公平比较 thinking 的真实能力，需要在**推理**（不是训练）时给足预算：

    python analyze/compare_thinking.py --model $M --thinking on \
        --max-tokens 16384 --problems 80 --n 4 --out results/th16k.json

代价：单卡约 6 小时。产出能回答「thinking 到底值不值这个预算」，
但不会改变训练侧的决定。

## 复现

    PATH=$RL_LAB/.tools/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 $VLLM_PY analyze/compare_thinking.py \
        --model $GRPO_BASELINE/execution/20261017/model \
        --thinking on --problems 80 --n 4 --max-tokens 6144 \
        --out results/th2_on.json

    # off 同理，改 --thinking off
