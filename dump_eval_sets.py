"""把评测数据导出成 JSONL，使评测端只需要 vLLM，不污染既有 vllm 环境。"""
import json
import os
from pathlib import Path

from datasets import load_dataset

R = Path(os.environ.get("GRPO_BASELINE_ROOT", "/path/to/rl-grpo8b-baseline"))
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

INSTR = json.loads((R / "baseline.json").read_text())["prompt_instruction"]


def strip_instruction(text):
    """去掉训练时用的 #### 指令，只留原题。"""
    if text.startswith(INSTR):
        return text[len(INSTR):].strip()
    marker = "Do not write anything after that line."
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def arc():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for r in ds:
        ch = r["choices"]
        body = "\n".join(f"{l}. {t}" for l, t in zip(ch["label"], ch["text"]))
        rows.append({
            "uid": r["id"],
            "user": ("Answer the following multiple-choice science question. "
                     "Reply with only the letter of the correct choice (A, B, C, or D).\n\n"
                     f"Question: {r['question']}\n{body}"),
            "gold": r["answerKey"],
        })
    return rows


def gsm_neutral():
    refs = [json.loads(l) for l in (R / "data/test.jsonl").read_text().split("\n") if l.strip()]
    rows = []
    for r in refs:
        raw = r["prompt"]
        content = raw[0]["content"] if isinstance(raw, list) else str(raw)
        question = strip_instruction(content)
        rows.append({
            "uid": r["extra_info"]["id"],
            "user": f"Solve this math problem. Give your final answer as a number.\n\n{question}",
            "gold": str(r["reward_model"]["ground_truth"]),
        })
    return rows


for name, rows in (("arc", arc()), ("gsm8k_neutral", gsm_neutral())):
    p = OUT / f"{name}.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    empty = sum(1 for r in rows if len(r["user"]) < 80)
    print(f"{name}: {len(rows)} 条, 疑似空题 {empty}")
    print("  样例:", rows[0]["user"][:220].replace("\n", " | "))
