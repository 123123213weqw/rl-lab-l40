"""量化"可学空间"：每个 prompt 采样 n 次，测 pass rate 分布。

关键量不是准确率，而是有多少 prompt 落在"有时对有时错"的区间——
只有这些 prompt 在 GRPO 里能产生非零优势。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import argparse
import json
import os
import re
from decimal import Decimal, InvalidOperation


R = Path(os.environ.get("GRPO_BASELINE", "/path/to/rl-grpo8b-baseline"))
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(s):
    try:
        return Decimal(str(s).replace(",", "").strip()).normalize()
    except (InvalidOperation, ArithmeticError):
        return None


def build(kind, limit):
    rows = []
    if kind == "gsm8k":
        refs = [json.loads(l) for l in (R / "data/test.jsonl").read_text().split("\n") if l.strip()]
        for r in refs[:limit]:
            raw = r["prompt"]
            content = raw[0]["content"] if isinstance(raw, list) else str(raw)
            rows.append({"uid": r["extra_info"]["id"], "user": content,
                         "gold": str(r["reward_model"]["ground_truth"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="gsm8k")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = build(args.kind, args.limit)
    tok = AutoTokenizer.from_pretrained(args.model)

    def chat(u):
        msgs = [{"role": "user", "content": u}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        except (TypeError, ValueError):
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096, seed=300,
              disable_log_stats=True)
    sp = SamplingParams(n=args.n, temperature=1.0, top_p=1.0,
                        max_tokens=args.max_tokens, seed=300)
    outs = llm.generate([chat(r["user"]) for r in rows], sp)

    recs = []
    for r, o in zip(rows, outs):
        gold = norm(r["gold"])
        ok = []
        for c in o.outputs:
            ns = [norm(m) for m in NUM.findall(c.text)]
            ns = [v for v in ns if v is not None]
            ok.append(int(bool(ns) and ns[-1] == gold))
        recs.append({"uid": r["uid"], "gold": r["gold"],
                     "n": len(ok), "correct": sum(ok),
                     "pass_rate": sum(ok) / len(ok)})

    Path(args.out).write_text(json.dumps(recs, indent=2))
    print(f"saved {args.out}  ({len(recs)} prompts x {args.n} samples)")
    print(f"整体准确率 {sum(r['correct'] for r in recs)/sum(r['n'] for r in recs):.4f}")


if __name__ == "__main__":
    main()
