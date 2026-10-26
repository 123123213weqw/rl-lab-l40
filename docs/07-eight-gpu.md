# 8 卡到底行不行

日期：2026-10-26。这份文档修正 `06-code-reward.md` 里一个算错的地方。

## 先承认：文档里那个数字是错的

`06-code-reward.md` 原写「8 卡全参：12 GB 参数 + 12 GB 梯度 + 24 GB Adam
≈ 48 GB，仍超」。这是把**总量**当成了**每卡量**。

正确算法（Qwen3-8B，8.19B 参数）：

    参数 bf16   8.19e9 x 2B = 16.4 GB
    梯度 bf16   16.4 GB
    Adam fp32   8.19e9 x 8B = 65.5 GB
    合计        98.3 GB

分片到 N 卡后每卡：98.3 / N。再加一份 vLLM 权重 16.4 GB。

| 卡数 | 模型状态/卡 | +vLLM | 小计 | 46 GB 余量 |
|---|---|---|---|---|
| 2 | 49.1 GB | 16.4 | 65.5 | -19.5 不可行 |
| 4 | 24.6 GB | 16.4 | 41.0 | +5.0 贴边 |
| **8** | **12.3 GB** | 16.4 | **28.7** | **+17.3 宽裕** |

4 卡算得 24.6 GB，与实测 24 GB 吻合，说明公式对。
**所以 8 卡在显存上不但可行，还比 4 卡宽裕得多。**

## 实测：8 卡确实跑通过一整步

一次成功的 8 卡运行（GPU 8 张各 7-8 GB 加载后）：

    step:1
    training/global_step: 1
    critic/score/mean: 0.0
    timing_s/step: 291.0
    actor/kl_loss: 0.0010

**8 卡跑完整了一步，291 秒。** 显存全程没有 OOM。

那一步 `score/mean = 0.0` 是因为 `response_length/clip_ratio = 1.0` ——
`max_response_length=1024` 把所有回复都截断了，代码没写完，沙箱全判 0。
这是我配置的问题，不是卡的问题。

## 但 8 卡现在稳定挂起，4 卡稳定可用

同一份代码和配置：

| 运行 | 卡 | 初始化 | 跑完 step:1 |
|---|---|---|---|
| smoke4test | 0,1,2,3 | 成功 | **成功**（470 s） |
| xtest | 0,1,4,5 | 成功 | 未等到 |
| smoke8 | 0-7 | **挂起** | 否 |

8 卡挂了 5 次以上，全部停在同一行：

    INFO: worker group kwargs: {device_name: cuda}

表现：GPU 各 3 MiB（只有 CUDA context），WorkerDict actor 一个都没创建，
Ray 显示 8 张 GPU 全空闲、无等待 lease、无 infeasible 队列。

排除掉的原因：

- **不是显存**（内存算过 28.7/46，且成功跑过一步）
- **不是 Ray 资源**（8 GPU available，CPU 充足）
- **不是 cgroup 限制**（无 cpu.max/memory.max，128 核可用）
- **不是残留 Ray 集群**（彻底 kill 后仍挂；日志里曾发现 3 个 gcs_server 并存）
- **不是 PCIe switch 跨组**（0,1,4,5 跨组 4 卡正常）
- **不是缺参数**（补齐到 84 个，含 `actor.strategy: fsdp`、
  `data.dataloader_num_workers: 0`、`lora_rank: 0`、clip/loss 细节，仍挂）
- **不是 CPU 配额**（24 -> 64 后第一次成功，之后又挂，说明只是巧合）

## 现状判断

8 卡**不是「不行」，是「不稳定」**：见过它成功，但现在复现不出来。
根因未定位，我不打算在没有证据的情况下继续猜。

可用的现状：

- **4 卡稳定**，每步 470 s
- **8 卡偶发挂起**，成功时每步 291 s（1.6x 快）

## 紧的约束其实是回复长度，不是卡数

4 卡那步的真实数据：

    critic/score/mean: 0.09375     <- dense 奖励生效，非零
    critic/score/max: 1.0          <- 有样本全过
    response_length/clip_ratio: 0.90625   <- 90.6% 被截断
    prompt_length/clip_ratio: 0.125

`max_response_length=2048` 仍然不够，90.6% 的代码没写完。这是当前
奖励几乎全零的主因，比换卡重要得多。代码题需要 3072-4096。

## 建议

1. **先用 4 卡把 max_response_length 提到 4096 跑通**，看 score/mean 能否上去。
   显存：4 卡模型状态 24.6 GB + vLLM，KV cache 需要装下 1536+4096 的序列。
   Qwen3-8B 的 KV 约 0.28 MB/token，5632 token 约 1.6 GB，可行。
2. 8 卡挂起问题单独查。可能方向：verl 的 `create_colocated_worker_cls`
   在 dp=8 时的路径、或某个 8 秩集合通信初始化。
3. 若 8 卡长期不稳，LoRA 是更稳的路线（模型状态降到约 6 GB/卡）。

## 复现

    cd $RL_LAB
    /tmp/cleanray.sh                     # 必须先清干净，否则残留 Ray 会干扰
    ./.venv-llm/bin/python launch_code_grpo.py --smoke --gpus 0,1,2,3
    # 换 0,1,2,3,4,5,6,7 则大概率挂在 worker group
