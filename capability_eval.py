"""在域外任务上比较 base 与 GRPO 训练后的 checkpoint。

  A. ARC-Challenge (1172)   —— 域外科学推理，检验通用能力是否被损伤
  B. GSM8K 换问法 (1319)     —— 不给 #### 指令，检验数学能力是否依赖训练出的格式

只依赖 vLLM + transformers；数据由 dump_eval_sets.py 预导出。
greedy 解码，seed=300。
"""
import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DATA = Path(__file__).resolve().parent / "data"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
LETTER = re.compile(r"\b([ABCD])\b")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().split("\n") if l.strip()]


def chat(tok, user):
    msgs = [{"role": "user", "content": user}]
    for kw in ({"enable_thinking": False}, {}):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        except (TypeError, ValueError):
            continue
    raise RuntimeError("chat template failed")


def norm(s):
    try:
        return Decimal(s.replace(",", "").strip()).normalize()
    except (InvalidOperation, AttributeError):
        return None


def score(kind, text, gold):
    if kind == "arc":
        m = LETTER.search(text.strip())
        return int(bool(m) and m.group(1) == gold), (m.group(1) if m else None)
    ns = [norm(x) for x in NUM.findall(text)]
    ns = [v for v in ns if v is not None]
    if not ns:
        return 0, None
    return int(ns[-1] == norm(gold)), str(ns[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-arc", type=int, default=0)   # 0 = 全量
    ap.add_argument("--n-gsm", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096, seed=300,
              disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=300)

    result = {"model": args.model, "suites": {}}
    for kind, path, limit in (("arc", DATA / "arc.jsonl", args.n_arc),
                              ("gsm8k_neutral", DATA / "gsm8k_neutral.jsonl", args.n_gsm)):
        rows = load_jsonl(path)
        if limit:
            rows = rows[:limit]
        prompts = [chat(tok, r["user"]) for r in rows]
        outs = llm.generate(prompts, sp)
        recs = []
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            ok, ext = score(kind, text, r["gold"])
            recs.append({"uid": r["uid"], "gold": r["gold"], "extracted": ext,
                         "ok": ok, "text": text})
        acc = sum(x["ok"] for x in recs) / len(recs)
        result["suites"][kind] = {"n": len(recs), "accuracy": acc, "records": recs}
        print(f"[{kind:14s}] n={len(recs):5d}  acc={acc:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
