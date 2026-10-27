# train

GRPO 训练。基于 veRL `main_ppo`。

| 文件 | 作用 |
|---|---|
| `code_reward.py` | 自定义奖励：按通过测试比例给分（dense） |
| `launch_code_grpo.py` | 生成 85 个 Hydra 参数并启动 |
| `run_probe.sh` | 复刻 baseline 的 99 个参数，做配置对照（920s→247s 就是它测的） |
| `run_sharded.sh` | 多卡分片跑 pass rate 测量 |
| `cleanray.sh` | 清理残留 Ray 集群（**每次启动前都要跑**） |

## dense 奖励

```
score = 通过测试数 / 总测试数       # 默认
score = 1.0 if 全过 else 0.0        # CODE_REWARD_MODE=binary
```

代码任务相对数学任务的核心优势：通过 15/20 个测试是可观测的中间信号，
而数学的二元对错只能给 0 或 1。

实测效果（300 题 × 8 采样）：零方差组 22.0% → 14.0%，
二元口径的零方差组里 36.4% 在 dense 下重获梯度。

环境变量：`CODE_REWARD_MODE`、`CODE_REWARD_TIMEOUT`、`CODE_REWARD_MAX_TESTS`、
`CODE_REWARD_SOURCE`。

## 必设的三项（漏了会静默失败）

1. **`+data.apply_chat_template_kwargs.enable_thinking=False`**
   veRL 默认 `{}`，Qwen3 chat template 就会开 thinking，思考过程吃光
   response 预算。漏掉的话训练"成功"但奖励恒为 0（见 `docs/08`）。
2. **`+ray_kwargs.ray_init.num_gpus=N`**
   不显式给，Ray 认为没有 GPU，所有 actor 挤到 GPU 0。
3. **`+ray_kwargs.ray_init.address=local`** 和独立 `_temp_dir`
   否则可能去连已有集群而挂住。

## 已验证的配置

4 卡，`max_response_length=4096`，`batch=8`，`G=8`（64 样本/步），
约 530 s/step，显存 30/46 GB。

8 卡显存充足（28.7/46 GB）且跑通过一步（291 s），但初始化会间歇性挂起，
根因未定位，见 `docs/07`。

## 奖励走的是 reward_loop 框架

不是旧的 `NaiveRewardManager.__call__`。`reward.num_workers` 决定创建多少个
RewardLoopWorker actor，框架层用 `asyncio.gather` 并发，每个 worker 内部
按 chunk 串行调 `run_single`。所以**不需要**自定义 reward manager。
