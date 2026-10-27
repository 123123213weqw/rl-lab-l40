"""用统一口径评测一个 checkpoint 在代码评测集上的表现。

输出两个口径，便于和训练日志、以及历史基线对齐：
  dense   平均 通过测试数/总测试数   —— 与训练时的 critic/score/mean 同口径
  binary  平均 是否全过              —— 与之前 GSM8K 那套二元口径对齐

必须显式传 --no-thinking，否则 Qwen3 会把预算耗在思考上（见 docs/08）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import argparse
import json
import re
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

FENCE = re.compile(r"```(?:python3?|py)?[ \t]*\n(.*?)```", re.S)
INSTRUCTION = (
    "You are an expert competitive programmer. Write a complete Python 3 program "
    "that reads from standard input and writes to standard output. "
    "Output only the code, no explanation."
)


def grade(job):
    i, j, code, tests, timeout = job
    from taco_lib import run_in_sandbox
    if not code:
        return i, j, 0, len(tests["inputs"]), False
    r = run_in_sandbox(code, tests, timeout=timeout)
    to = "no_result" in (r.get("errors") or [])
    return i, j, int(r.get("passed", 0)), int(r.get("total", 0)), to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/eval_code.jsonl"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=3)
    ap.add_argument("--graders", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in Path(args.data).read_text().split("\n") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print("eval 题数 %d, n=%d, t=%.2f" % (len(rows), args.n, args.temperature), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": INSTRUCTION + "\n\nProblem:\n" + r["question"].strip()}]
        try:
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        except (TypeError, ValueError):
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096, seed=311,
              disable_log_stats=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, seed=311)
    outs = llm.generate(prompts, sp)
    print("生成完成", flush=True)

    jobs = []
    for i, (r, o) in enumerate(zip(rows, outs)):
        for j, c in enumerate(o.outputs):
            blocks = FENCE.findall(c.text)
            code = (blocks[-1] if blocks else c.text).strip() or None
            jobs.append((i, j, code, r["tests"], args.timeout))

    per = {}
    n_to = 0
    with ProcessPoolExecutor(max_workers=args.graders) as ex:
        futs = [ex.submit(grade, j) for j in jobs]
        for f in as_completed(futs):
            i, j, p, t, to = f.result()
            per[(i, j)] = (p, t)
            n_to += int(to)

    ratios, binaries, lens = [], [], []
    for i in range(len(rows)):
        for j in range(args.n):
            p, t = per.get((i, j), (0, 0))
            ratios.append(p / t if t else 0.0)
            binaries.append(1.0 if 0 < p == t else 0.0)
    for o in outs:
        lens += [len(c.token_ids) for c in o.outputs]

    print()
    print("=" * 56)
    print("dense  平均通过比例: %.4f" % statistics.mean(ratios))
    print("binary 平均全过比例: %.4f" % statistics.mean(binaries))
    print("平均输出长度: %.1f token" % statistics.mean(lens))
    print("超时样本: %d / %d (%.2f%%)" % (n_to, len(jobs), 100 * n_to / len(jobs)))
    print("=" * 56)

    # 逐题的 dense/binary，供配对 bootstrap 用
    per_problem = []
    for i, r in enumerate(rows):
        rs = [per.get((i, j), (0, 0)) for j in range(args.n)]
        pr = [(p / t if t else 0.0) for p, t in rs]
        pb = [1.0 if 0 < p == t else 0.0 for p, t in rs]
        per_problem.append({"id": r["id"], "dense": sum(pr) / len(pr),
                            "binary": sum(pb) / len(pb),
                            "difficulty": r.get("difficulty")})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": args.model, "n": args.n, "temperature": args.temperature,
        "dense": statistics.mean(ratios), "binary": statistics.mean(binaries),
        "mean_len": statistics.mean(lens), "timeout_rate": n_to / len(jobs),
        "n_problems": len(rows), "per_problem": per_problem,
    }, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
