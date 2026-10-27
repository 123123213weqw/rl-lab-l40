"""诊断：dense 奖励是否真的带来可用的组内方差？

二元奖励的问题不是"给 0 分不对"，而是组内所有样本同分 -> 优势恒为零。
本脚本用真实模型采样，对同一批样本同时算两种口径，逐组比较。

输出：
  - 二元口径下零方差的组，在 dense 口径下有多少变成有方差
  - 优势值的实际分布（±1.5 之类的 GRPO 标准化后）
  - 超时率（决定 reward 阶段耗时）

用法:
  python measure_reward_density.py --model <ckpt> --problems 300 --n 8 --out ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

FENCE = re.compile(r"```(?:python3?|py)?[ \t]*\n(.*?)```", re.S)

SYSTEM = (
    "You are an expert competitive programmer. Write a complete Python 3 program "
    "that reads from standard input and writes to standard output. "
    "Output only the code, no explanation."
)


def extract_code(text):
    blocks = FENCE.findall(text)
    return (blocks[-1] if blocks else text).strip() or None


def grade(job):
    i, j, code, tests, timeout = job
    from taco_lib import run_in_sandbox
    if code is None:
        return i, j, 0, len(tests["inputs"]), False
    r = run_in_sandbox(code, tests, timeout=timeout)
    timed_out = "no_result" in (r.get("errors") or [])
    return i, j, int(r.get("passed", 0)), int(r.get("total", 0)), timed_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/train_code.jsonl"))
    ap.add_argument("--problems", type=int, default=300)
    ap.add_argument("--n", type=int, default=8, help="组大小 G")
    ap.add_argument("--timeout", type=int, default=3)
    ap.add_argument("--graders", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in Path(args.data).read_text().split("\n") if l.strip()]
    rows = rows[: args.problems]
    print("题目数: %d, 每题采样 %d" % (len(rows), args.n), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)

    def chat(q):
        msgs = [{"role": "user", "content": SYSTEM + "\n\nProblem:\n" + q.strip()}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except (TypeError, ValueError):
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=6144, seed=311,
              disable_log_stats=True)
    sp = SamplingParams(n=args.n, temperature=0.8, top_p=0.95,
                        max_tokens=args.max_tokens, seed=311)
    outs = llm.generate([chat(r["question"]) for r in rows], sp)
    print("生成完成", flush=True)

    jobs = []
    for i, (r, o) in enumerate(zip(rows, outs)):
        for j, c in enumerate(o.outputs):
            jobs.append((i, j, extract_code(c.text), r["tests"], args.timeout))

    per = defaultdict(dict)
    n_timeout = 0
    done = 0
    import time
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.graders) as ex:
        futs = {ex.submit(grade, j): (j[0], j[1]) for j in jobs}
        for f in as_completed(futs):
            i, j, p, t, to = f.result()
            per[i][j] = (p, t)
            n_timeout += int(to)
            done += 1
            if done % 500 == 0:
                print("  评测 %d/%d  %.1fs" % (done, len(jobs), time.time() - t0), flush=True)
    grade_sec = time.time() - t0

    # 逐组比较两种口径
    groups = []
    for i, r in enumerate(rows):
        ratios = []
        for j in range(args.n):
            p, t = per.get(i, {}).get(j, (0, 0))
            ratios.append(p / t if t else 0.0)
        binary = [1.0 if x >= 1.0 and x > 0 else 0.0 for x in ratios]
        groups.append({
            "id": r["id"], "difficulty": r["difficulty"],
            "ratios": ratios, "binary": binary,
            "mean_ratio": sum(ratios) / len(ratios),
            "mean_binary": sum(binary) / len(binary),
        })

    def zero_var(vals):
        return max(vals) == min(vals)

    n_groups = len(groups)
    zv_bin = sum(1 for g in groups if zero_var(g["binary"]))
    zv_dense = sum(1 for g in groups if zero_var(g["ratios"]))
    rescued = sum(1 for g in groups
                  if zero_var(g["binary"]) and not zero_var(g["ratios"]))

    print()
    print("=" * 62)
    print("组内方差对比（G=%d，%d 组）" % (args.n, n_groups))
    print("=" * 62)
    print("  二元口径零方差组: %4d / %d = %5.1f%%" % (zv_bin, n_groups, 100 * zv_bin / n_groups))
    print("  dense 口径零方差组: %4d / %d = %5.1f%%" % (zv_dense, n_groups, 100 * zv_dense / n_groups))
    print("  二元零方差 -> dense 有方差（被救回）: %d 组" % rescued)
    print()
    resc = 100 * rescued / zv_bin if zv_bin else 0
    print("  二元口径的零方差里，%.1f%% 在 dense 下重新获得梯度" % resc)
    print()
    print("  有效组比例: 二元 %.1f%%  ->  dense %.1f%%"
          % (100 * (1 - zv_bin / n_groups), 100 * (1 - zv_dense / n_groups)))

    # 超时率
    print()
    print("耗时与超时")
    print("  评测样本数: %d" % len(jobs))
    print("  沙箱总耗时: %.1fs（%d 并发）" % (grade_sec, args.graders))
    print("  单样本串行等效: %.3fs" % (grade_sec * args.graders / len(jobs)))
    print("  超时样本: %d (%.2f%%)" % (n_timeout, 100 * n_timeout / len(jobs)))

    # 预测训练步开销（串行评分）
    per_sample = grade_sec * args.graders / len(jobs)
    for bs, g in ((32, 4), (32, 8)):
        ns = bs * g
        print("  batch=%d G=%d -> %d 样本, 串行评分约 %.1fs" % (bs, g, ns, ns * per_sample))

    # dense 的优势分布
    print()
    print("标准化后的优势分布（GRPO：减组内均值除标准差）")
    for name, key in (("binary", "binary"), ("dense", "ratios")):
        advs = []
        for g in groups:
            v = g[key]
            if max(v) == min(v):
                continue
            m = sum(v) / len(v)
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            if sd == 0:
                continue
            advs.extend([(x - m) / sd for x in v])
        if advs:
            print("  %-6s n=%5d  mean=%+.3f  sd=%.3f  min=%+.2f  max=%+.2f"
                  % (name, len(advs), sum(advs) / len(advs), statistics.stdev(advs),
                     min(advs), max(advs)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "groups": groups, "n_per_group": args.n,
        "zero_var_binary": zv_bin, "zero_var_dense": zv_dense,
        "rescued": rescued, "n_groups": n_groups,
        "timeout_rate": n_timeout / len(jobs),
        "per_sample_serial_sec": per_sample,
    }, indent=2))
    print()
    print("saved", out)


if __name__ == "__main__":
    main()
