"""测 MATH-500 分难度层的 pass rate 分布，并与 GSM8K 对照。

直接回答：更难的数据到底有没有可学空间，还是只是从"太简单"变成"太难"。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import argparse
import json
import re
from collections import Counter, defaultdict


BOX = re.compile(r"\\boxed\s*\{")
ANS_LINE = re.compile(r"(?:final answer is|answer is|答案是)\s*[:：]?\s*(.+)")


def extract_boxed(text):
    """取最后一个 \\boxed{...}，支持嵌套花括号。"""
    starts = [m.end() for m in BOX.finditer(text)]
    for start in reversed(starts):
        depth, i = 1, start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i - 1]
    return None


def norm(s):
    if s is None:
        return None
    s = str(s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("$", "").replace(" ", "").replace("\n", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = s.rstrip(".")
    return s


def cand(text):
    b = extract_boxed(text)
    if b is not None:
        return norm(b)
    m = ANS_LINE.search(text)
    if m:
        return norm(m.group(1).strip())
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return norm(nums[-1]) if nums else None


def correct(pred, gold):
    p, g = norm(pred), norm(gold)
    if p is None or g is None:
        return 0
    if p == g:
        return 1
    # 数值等价
    try:
        from fractions import Fraction
        if Fraction(p.replace(",", "")) == Fraction(g.replace(",", "")):
            return 1
    except Exception:
        pass
    return 0




def build(path, limit_per_level):
    rows_all = [json.loads(l) for l in (DATA / path).read_text().split("\n") if l.strip()]
    if not limit_per_level:
        return rows_all
    by_level = defaultdict(list)
    for r in rows_all:
        by_level[r["level"]].append(r)
    rows = []
    for lvl in sorted(by_level):
        rows.extend(by_level[lvl][:limit_per_level])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="math500.jsonl")
    ap.add_argument("--per-level", type=int, default=48)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = build(args.data, args.per_level)
    print(f"{len(rows)} 题, 每层 {args.per_level}, 每题 {args.n} 采样", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    def chat(u):
        msgs = [{"role": "user", "content": u}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except (TypeError, ValueError):
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=5120, seed=300,
              disable_log_stats=True)
    sp = SamplingParams(n=args.n, temperature=1.0, top_p=1.0,
                        max_tokens=args.max_tokens, seed=300)
    outs = llm.generate([chat(r["user"]) for r in rows], sp)

    recs = []
    for r, o in zip(rows, outs):
        ok = [correct(cand(c.text), r["gold"]) for c in o.outputs]
        noext = sum(1 for c in o.outputs if cand(c.text) is None)
        recs.append({**{k: r[k] for k in ("uid", "level", "subject", "gold")},
                     "n": len(ok), "correct": sum(ok),
                     "pass_rate": sum(ok) / len(ok), "no_extract": noext})
    Path(args.out).write_text(json.dumps(recs, indent=2))

    print(f"\n整体准确率 {sum(r['correct'] for r in recs)/sum(r['n'] for r in recs):.4f}")
    print(f"\n{'level':>6} {'n题':>5} {'准确率':>8} {'全对%':>7} {'全错%':>7} {'有效带%':>8}")
    for lvl in sorted({r["level"] for r in recs}):
        g = [r for r in recs if r["level"] == lvl]
        acc = sum(r["correct"] for r in g) / sum(r["n"] for r in g)
        full = sum(1 for r in g if r["correct"] == r["n"]) / len(g)
        zero = sum(1 for r in g if r["correct"] == 0) / len(g)
        band = sum(1 for r in g if 0.1 <= r["pass_rate"] <= 0.9) / len(g)
        print(f"{lvl:>6} {len(g):>5} {acc:>8.3f} {full:>6.1%} {zero:>6.1%} {band:>7.1%}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
