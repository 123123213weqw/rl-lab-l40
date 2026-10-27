"""对比 thinking 开/关：不只是长度，看实际输出内容和能力差。

输出：
  - pass rate（dense / binary）
  - 长度分布
  - 前若干条原始文本，供人眼判断有没有推理
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
THINK = re.compile("</" + "think" + ">", re.S)
INSTRUCTION = (
    "You are an expert competitive programmer. Write a complete Python 3 program "
    "that reads from standard input and writes to standard output. "
    "Output only the code, no explanation."
)


def grade(job):
    i, j, code, tests, timeout = job
    from taco_lib import run_in_sandbox
    if not code:
        return i, j, 0, len(tests["inputs"])
    r = run_in_sandbox(code, tests, timeout=timeout)
    return i, j, int(r.get("passed", 0)), int(r.get("total", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/eval_code.jsonl"))
    ap.add_argument("--thinking", choices=["on", "off"], required=True)
    ap.add_argument("--problems", type=int, default=150)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=int, default=3)
    ap.add_argument("--graders", type=int, default=40)
    ap.add_argument("--dump", type=int, default=3, help="dump 几条原文")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in Path(args.data).read_text().split("\n") if l.strip()]
    rows = rows[: args.problems]
    think = args.thinking == "on"
    print("%s: %d 题 x %d, thinking=%s, max_tokens=%d"
          % (args.model.split("/")[-1], len(rows), args.n, args.thinking, args.max_tokens),
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": INSTRUCTION + "\n\nProblem:\n" + r["question"].strip()}]
        try:
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think))
        except (TypeError, ValueError) as e:
            print("  chat template 不支持 enable_thinking=%s: %s" % (think, e))
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.85,
              max_model_len=args.max_tokens + 2048, seed=311, disable_log_stats=True)
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
    with ProcessPoolExecutor(max_workers=args.graders) as ex:
        for f in as_completed([ex.submit(grade, j) for j in jobs]):
            i, j, p, t = f.result()
            per[(i, j)] = (p, t)

    ratios, bins, lens, ntrunc, nthink = [], [], [], 0, 0
    for i in range(len(rows)):
        for j in range(args.n):
            p, t = per.get((i, j), (0, 0))
            ratios.append(p / t if t else 0.0)
            bins.append(1.0 if 0 < p == t else 0.0)
    for o in outs:
        for c in o.outputs:
            lens.append(len(c.token_ids))
            if len(c.token_ids) >= args.max_tokens:
                ntrunc += 1
            if THINK.search(c.text):
                nthink += 1

    print()
    print("=" * 58)
    print("thinking=%s  dense=%.4f  binary=%.4f"
          % (args.thinking, statistics.mean(ratios), statistics.mean(bins)))
    print("输出长度: 均 %.0f  中位 %.0f  max %d"
          % (statistics.mean(lens), statistics.median(lens), max(lens)))
    print("含  标记的比例: %.1f%%" % (100 * nthink / len(lens)))
    print("撞到 max_tokens 的比例: %.1f%%" % (100 * ntrunc / len(lens)))
    print("=" * 58)

    if args.dump:
        print()
        print("=== 原文样本（前 %d 条，各取 600 字符）===" % args.dump)
        for i in range(min(args.dump, len(outs))):
            t = outs[i].outputs[0].text
            print("--- %s (len=%d) ---" % (rows[i]["id"], len(outs[i].outputs[0].token_ids)))
            print(t[:600].replace("\n", "\n  "))
            print()

    Path(args.out).write_text(json.dumps({
        "model": args.model, "thinking": args.thinking, "n": args.n,
        "max_tokens": args.max_tokens, "problems": len(rows),
        "dense": statistics.mean(ratios), "binary": statistics.mean(bins),
        "mean_len": statistics.mean(lens), "median_len": statistics.median(lens),
        "trunc_rate": ntrunc / len(lens),
        "think_tag_rate": nthink / len(lens),
        "per_problem": [{"id": rows[i]["id"],
                         "dense": sum(per.get((i, j), (0, 0))[0] / max(per.get((i, j), (0, 1))[1], 1)
                                      for j in range(args.n)) / args.n}
                        for i in range(len(rows))],
    }, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
