"""阶段 2: 用基座模型对可信题目采样，测 pass rate。

这是决定数据集可用性的关键一步——不看准确率，看有多少题落在
"有时对有时错"的可学区间。

用法:
  python grade_candidates.py --validated results/taco_validated.json \
      --model <ckpt> --n 8 --limit 400 --out results/code_passrate.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import argparse
import json
import re


CACHE = str(ROOT / "hf-cache")

SYSTEM = (
    "You are an expert competitive programmer. Write a complete Python 3 program "
    "that reads from standard input and writes to standard output. "
    "Output only the code, no explanation."
)

# 从模型输出中抽出代码块
FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    m = FENCE.search(text)
    if m:
        return m.group(1).strip()
    # 没有围栏就当整段是代码
    return text.strip()


def build_prompt(question):
    return "{}\n\nProblem:\n{}".format(SYSTEM, question.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--graders", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = json.loads(Path(args.validated).read_text())
    if args.limit:
        rows = rows[: args.limit]
    print("待测题目:", len(rows), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)

    def chat(q):
        msgs = [{"role": "user", "content": build_prompt(q)}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except (TypeError, ValueError):
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=6144, seed=300,
              disable_log_stats=True)
    sp = SamplingParams(n=args.n, temperature=0.8, top_p=0.95,
                        max_tokens=args.max_tokens, seed=300)
    outs = llm.generate([chat(r["question"]) for r in rows], sp)
    print("生成完成，开始沙箱评测", flush=True)

    # 评测阶段用进程池，避免 GPU 与大量子进程争抢
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from taco_lib import run_in_sandbox

    jobs = []
    for i, (r, o) in enumerate(zip(rows, outs)):
        for j, c in enumerate(o.outputs):
            code = extract_code(c.text)
            jobs.append((i, j, code, r["tests"]))

    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.graders) as ex:
        futs = {ex.submit(_grade, j): (j[0], j[1]) for j in jobs}
        for f in as_completed(futs):
            i, j, passed, total = f.result()
            results.setdefault(i, {})[j] = (passed, total)
            done += 1
            if done % 200 == 0:
                print("  评测 %d/%d" % (done, len(jobs)), flush=True)

    recs = []
    for i, r in enumerate(rows):
        per = results.get(i, {})
        ok = sum(1 for j in range(len(outs[i].outputs)) if per.get(j, (0, 1))[0] == per.get(j, (0, 1))[1] and per.get(j, (0, 0))[1] > 0)
        n = len(outs[i].outputs)
        recs.append({
            "idx": r.get("idx", r.get("id")), "difficulty": r["difficulty"],
            "source": r["source"],
            "n": n, "correct": ok, "pass_rate": ok / n if n else 0.0,
            "n_tests": r["n_tests"], "url": r.get("url"),
        })
    Path(args.out).write_text(json.dumps(recs, indent=2))

    tot = sum(r["correct"] for r in recs)
    den = sum(r["n"] for r in recs)
    print()
    print("整体 pass@1 估计: %.4f  (%d/%d)" % (tot / den, tot, den))
    from collections import Counter
    hist = Counter(r["correct"] for r in recs)
    print()
    print("每 prompt 通过样本数（共 %d 个）:" % args.n)
    for k in range(args.n + 1):
        c = hist.get(k, 0)
        print("  %2d/%d : %4d  (%5.1f%%)" % (k, args.n, c, 100.0 * c / len(recs)))
    print()
    print("saved", args.out)


def _grade(job):
    i, j, code, tests = job
    from taco_lib import run_in_sandbox
    r = run_in_sandbox(code, tests)
    return i, j, r["passed"], r["total"]


if __name__ == "__main__":
    main()
