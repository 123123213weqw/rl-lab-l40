# 环境与硬件

建立日期：2026-10-26。目录：`$RL_LAB/`

## 环境

- 解释器：`$RL_LAB/.venv/bin/python`（Python 3.10.12）
- 用 `--system-site-packages` 创建，复用系统已有的 torch 2.14.1+cu130
- 已装：gymnasium 1.4.0、stable-baselines3 2.9.0、tensorboard 2.21.0、ale-py、opencv、pygame-ce

## 关键约束：GPU 全部被占

2026-10-26 检查时，8 × L40 被本账号的 vLLM（GLM-5.3-Flash W4A16, TP=8,
`--gpu-memory-utilization 0.975`, 1M context, 127.0.0.1:18550）占满，
**每卡仅剩 168–208 MiB 空闲**，连一个 CUDA context 都建不起来。

后果：任何 PyTorch 代码都会崩，哪怕写了 `device="cpu"`。
因为 torch 2.14 的 `Adam.step()` 内部会调 `torch.accelerator.current_stream()`
做健康检查，这会触发 CUDA 初始化 → `cudaErrorMemoryAllocation`。

对策（脚本里已内置）：
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 必须在 import torch 之前
```
或者运行时：`CUDA_VISIBLE_DEVICES= ./.venv/bin/python xxx.py`

## 脚本

| 文件 | 作用 |
|---|---|
| `00_sanity.py` | 摸清 env 五元组接口；随机策略 ≈ 21 分 |
| `01_cartpole_ppo.py` | PPO 训 CartPole，60k 步，约 25 秒（CPU 8 并行环境） |
| `02_multiseed_curve.py` | 3 种子 × 50k 步学习曲线；**正确评估姿势的示范** |

## 已确认结果

`02_multiseed_curve.py`（10 局确定性评估的均值）：

```
种子 |      0步 |  10000步 |  20000步 |  30000步 |  40000步 |  50000步
   0 |   384.1 |   104.2 |   436.6 |   291.1 |   496.9 |   346.6
   1 |     8.8 |   124.5 |   234.8 |   289.7 |   242.5 |   401.3
   2 |     9.0 |    94.9 |   469.6 |   496.6 |   439.9 |   500.0
均值 |   134.0 |   107.9 |   380.3 |   359.1 |   393.1 |   416.0
```

三个可观察现象：
1. **0 步方差极大**（9 → 384）：未训练策略的水平全靠种子抽奖，
   所以"训练前 X → 训练后 Y"单点对比不能当证据。
2. **初期性能会掉**（10k 步均值 108 < 初始 134）：策略被推离随机初值，
   同时价值函数还没学准，是 PPO 的常见现象，不是 bug。
3. **真正收敛到接近满分**（40k–50k 步，最好种子满分 500）。

## 机器规格（2026-10-26）

- Ubuntu 22.04.2, kernel 5.15.0-190
- 2 × Xeon Gold 6430：64 核 / 128 线程，503 GB RAM（可用 ~438 GB）
- 8 × NVIDIA L40 46 GB，驱动 580.126.09，CUDA 13.0
- 磁盘：`/data1` 357G 可用、`/data` 2.5T 可用、`/myjfs` 6T 可用、`/` 183G 可用
- 网络：pypi 直连 2.0s，清华镜像 1.3s，github 0.7s（无需代理）
- EGL/GL 库齐全（MuJoCo 无头渲染可用）；未装 ffmpeg

## 注意

- 该节点为多人共享，不要操作其他用户的进程。
- 大文件放 `/data1`、`/data` 或 `/myjfs`，不要写 `/home`（系统盘只剩 183G）。
- 纯 CPU RL：128 线程可以靠 `n_envs=8~32` 的向量化环境把吞吐拉起来。
