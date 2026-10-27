"""导出 MATH-500 与 AIME 2024 为 JSONL（评测端只需 vLLM）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import json
from collections import defaultdict

from datasets import load_dataset


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    by_level = defaultdict(list)
    for r in ds:
        by_level[r["level"]].append(r)
    print("MATH-500 分层计数:", {k: len(v) for k, v in sorted(by_level.items())})

    rows = []
    for lvl in sorted(by_level):
        for r in by_level[lvl]:
            rows.append({
                "uid": r["unique_id"], "level": r["level"], "subject": r["subject"],
                "gold": r["answer"],
                "user": ("Solve the following math problem. Put your final answer in \\boxed{}.\n\n"
                         + r["problem"]),
            })
    (DATA / "math500.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"math500: {len(rows)} 条")

    aime = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    arows = []
    for r in aime:
        arows.append({
            "uid": str(r["ID"]), "level": "AIME", "subject": "AIME",
            "gold": str(r["Answer"]),
            "user": ("Solve the following math problem. Put your final answer in \\boxed{}.\n\n"
                     + r["Problem"]),
        })
    (DATA / "aime2024.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in arows) + "\n")
    print(f"aime2024: {len(arows)} 条")
    print("样例 gold:", rows[0]["gold"], "|", arows[0]["gold"])


if __name__ == "__main__":
    main()
