#!/usr/bin/env python3
"""启动代码任务的 GRPO 训练。

与 GSM8K baseline 的关键差异
----------------------------
1. 数据: data/verl/{train,test}.parquet（TACO 过滤后，1333/326 题）
2. 奖励: code_reward.compute_score，按通过的测试比例给分（dense）
3. reward manager: ParallelRewardManager（并发评分，比 naive 快 6x）
4. 训练配置沿用已验证的提速设置（关 offload、micro_batch=4）

用法
----
  # 冒烟测试（1 步，验证接线）
  python launch_code_grpo.py --smoke

  # 正式跑 50 步，4 卡
  python launch_code_grpo.py --gpus 0,1,2,3 --steps 50

  # binary 奖励对照（复现原口径）
  CODE_REWARD_MODE=binary python launch_code_grpo.py --steps 50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import argparse
import json
import os
import subprocess


# 用环境变量指定，避免把本机路径写进仓库
#   GRPO_BASELINE=/path/to/grpo-baseline
_BASE = os.environ.get("GRPO_BASELINE", "")
DEFAULT_VERL = Path(_BASE) / "vendor/verl" if _BASE else Path("/nonexistent/vendor/verl")
DEFAULT_MODEL = Path(_BASE) / "execution/20261017/model" if _BASE else Path("/nonexistent/model")

# 已验证的提速配置：关 offload 后 920s -> 247s/step，MFU 0.40% -> 1.88%
FAST_FSDP = {
    "actor_rollout_ref.actor.fsdp_config.param_offload": False,
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
}


def build(
    verl_root, model, train_file, val_file, out_dir,
    gpus, steps, group_size, batch_size, micro_batch,
    lr, kl_coef, temperature, max_tokens, seed, reward_workers, smoke, save_freq,
):
    python = str(verl_root / ".venv/bin/python")

    values = {
        # ---- 算法: GRPO ----
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,          # KL 放在 loss 里
        "algorithm.kl_ctrl.kl_coef": kl_coef,
        "algorithm.filter_groups.enable": False,
        "algorithm.norm_adv_by_std_in_grpo": True,

        # ---- 数据 ----
        "data.train_files": str(train_file),
        "data.val_files": str(val_file),
        "data.train_batch_size": batch_size,
        "data.val_batch_size": 8,
        "data.max_prompt_length": 1536,
        "data.max_response_length": max_tokens,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.dataloader_num_workers": 0,
        "data.seed": seed,
        # 必须显式关闭 thinking：Qwen3 系列 chat template 默认开启，
        # 打开时思考过程会吃掉全部 response 预算，代码写不完 ->
        # 全部截断 -> 奖励恒为 0。baseline 一直是这么设的。
        "+data.apply_chat_template_kwargs.enable_thinking": False,

        # ---- 模型 ----
        "actor_rollout_ref.model.path": str(model),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.trust_remote_code": False,
        "actor_rollout_ref.model.lora_rank": 0,
        "actor_rollout_ref.model.lora.rank": 0,
        "actor_rollout_ref.model.lora_adapter_path": None,

        # ---- actor ----
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.optim.lr": lr,
        "actor_rollout_ref.actor.ppo_mini_batch_size": batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": micro_batch,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": kl_coef,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.clip_ratio": 0.2,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.2,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "bfloat16",
        "actor_rollout_ref.actor.fsdp_config.dtype": "bfloat16",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.actor.fsdp_config.seed": seed,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,

        # ---- rollout ----
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.55,
        "actor_rollout_ref.rollout.n": group_size,
        "actor_rollout_ref.rollout.temperature": temperature,
        "actor_rollout_ref.rollout.top_p": 0.95,
        "actor_rollout_ref.rollout.seed": seed,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 2048,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 2,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,

        # ---- ref ----
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 2,
        "actor_rollout_ref.ref.fsdp_config.param_offload": False,
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "bfloat16",
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile": False,

        # ---- 奖励: 自定义函数 + 并行 manager ----
        # 注意：这是给 veRL 的**路径字符串**，不是 import，
        # 移动文件时必须同步改这里（reward 模块加载失败会导致奖励恒 0）。
        "reward.custom_reward_function.path": str(ROOT / "train" / "code_reward.py"),
        "reward.custom_reward_function.name": "compute_score",
        # 这个 veRL 版本走 reward_loop 框架：num_workers 个 RewardLoopWorker
        # 由 asyncio.gather 并发，每个 worker 内部按 chunk 串行跑 run_single。
        # 所以并发度靠 num_workers，不需要自定义 manager。
        "reward.num_workers": reward_workers,
        "reward.reward_model.enable": False,
        "reward.reward_manager.source": "register",
        "reward.reward_manager.name": "naive",

        # ---- trainer ----
        "trainer.logger": ["console", "tensorboard"],
        "trainer.project_name": "grpo_code",
        "trainer.experiment_name": "code_dense_grpo",
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": len(gpus),
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": 1 if smoke else steps,
        "trainer.save_freq": -1 if smoke else save_freq,
        "trainer.test_freq": -1,
        "trainer.val_before_train": False,
        "trainer.resume_mode": "disable",
        "trainer.default_local_dir": str(out_dir / "checkpoints"),

        # ---- ray ----
        "ray_kwargs.ray_init.num_cpus": 24,
        "+ray_kwargs.ray_init.num_gpus": len(gpus),
        # 必须显式 local + 独立 temp_dir，否则可能去连已有集群而挂住
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init.include_dashboard": False,
        "+ray_kwargs.ray_init._temp_dir": str(out_dir / "ray"),
        "ray_kwargs.ray_init.runtime_env.py_executable": python,
    }
    values.update(FAST_FSDP)

    args = ["%s=%s" % (k, json.dumps(v, separators=(",", ":")))
            for k, v in values.items()]
    cmd = [python, "-m", "verl.trainer.main_ppo"] + args

    env = dict(os.environ)
    env["PYTHONPATH"] = str(verl_root)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    # 沙箱需要 ninja 在 PATH；vllm 的 flashinfer 在此机器上不可用
    env["PATH"] = str(ROOT / ".tools/bin") + ":" + env.get("PATH", "")
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    # 奖励函数读这些
    env.setdefault("CODE_REWARD_MODE", "dense")
    env.setdefault("CODE_REWARD_TIMEOUT", "3")
    env["CODE_REWARD_WORKERS"] = str(reward_workers)
    return cmd, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verl", default=str(DEFAULT_VERL))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--train", default=str(ROOT / "data/verl/train.parquet"))
    ap.add_argument("--val", default=str(ROOT / "data/verl/test.parquet"))
    ap.add_argument("--out", default=str(ROOT / "runs/code-grpo"))
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl-coef", type=float, default=0.001)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=311)
    ap.add_argument("--reward-workers", type=int, default=2)
    ap.add_argument("--save-freq", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = ap.parse_args()

    verl_root = Path(args.verl)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 前置校验：任一缺失都会让训练在几十分钟后静默失败或奖励恒为 0
    reward_mod = ROOT / "train" / "code_reward.py"
    sandbox_dir = ROOT / "sandbox"
    for p in (verl_root / ".venv/bin/python", Path(args.model), Path(args.train),
              Path(args.val), reward_mod):
        if not p.exists():
            sys.exit("缺少: %s" % p)
    for p in (sandbox_dir / "taco_lib.py", sandbox_dir / "driver.py",
              sandbox_dir / "sandbox_run.sh"):
        if not p.exists():
            sys.exit("沙箱文件缺失: %s（奖励会在运行时抛错，导致奖励恒 0）" % p)

    cmd, env = build(
        verl_root, Path(args.model), Path(args.train), Path(args.val), out,
        args.gpus.split(","), args.steps, args.group_size, args.batch_size,
        args.micro_batch, args.lr, args.kl_coef, args.temperature,
        args.max_tokens, args.seed, args.reward_workers, args.smoke, args.save_freq,
    )

    print("奖励模式:", env["CODE_REWARD_MODE"],
          "| 超时:", env["CODE_REWARD_TIMEOUT"], "s",
          "| 评分并发:", args.reward_workers)
    print("样本/步:", args.batch_size * args.group_size)
    print()

    if args.dry_run:
        print(" ".join(cmd[:6]), "...  (共 %d 个参数)" % (len(cmd) - 3))
        return

    log = out / "train.log"
    print("日志:", log)
    with log.open("w") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    print("退出码:", proc.returncode)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
