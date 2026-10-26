"""阶段 3: 从已验证题目构建代码数据集。

做四件事:
  1. 丢弃测试不可信/不稳定的来源（geeksforgeeks、leetcode 等）
  2. 按题干归一化去重
  3. 按题目 id 的哈希做确定性切分，保证 train/eval 不相交
  4. 输出 JSONL
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

# 全量验证中可信率极低（<30%）的来源，其测试数据在 TACO 中本身是坏的
BAD_SOURCES = {"geeksforgeeks", "leetcode", "kattis", "hackerearth"}

WS = re.compile(r"\s+")


def qhash(question):
    """题干归一化后取哈希，用于去重和切分。"""
    t = WS.sub(" ", question.strip().lower())
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def split_of(h, eval_frac=0.10, dev_frac=0.05):
    """用哈希前缀做确定性切分，避免随机种子漂移，且天然不相交。"""
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < eval_frac:
        return "eval"
    if v < eval_frac + dev_frac:
        return "dev"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated", default="results/taco_validated.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-bad-sources", action="store_true", default=True)
    ap.add_argument("--keep-bad-sources", dest="drop_bad_sources", action="store_false")
    args = ap.parse_args()

    rows = json.loads(Path(args.validated).read_text())
    print("验证通过的题目:", len(rows))

    kept, dropped_src, seen = [], Counter(), set()
    for r in rows:
        if args.drop_bad_sources and r["source"] in BAD_SOURCES:
            dropped_src[r["source"]] += 1
            continue
        h = qhash(r["question"])
        if h in seen:
            dropped_src["duplicate"] += 1
            continue
        seen.add(h)
        r["qhash"] = h
        r["split"] = split_of(h)
        # 单题测试用例过多会拖慢评分，截断到 20 个
        t = r["tests"]
        n = min(len(t["inputs"]), 20)
        r["tests"] = {"inputs": t["inputs"][:n], "outputs": t["outputs"][:n]}
        r["n_tests"] = n
        kept.append(r)

    print("剔除:", dict(dropped_src))
    print("保留:", len(kept))

    print()
    print("保留后的难度分布:", Counter(r["difficulty"] for r in kept).most_common())
    print("保留后的来源分布:", Counter(r["source"] for r in kept).most_common())
    print("切分:", Counter(r["split"] for r in kept).most_common())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in kept:
            rec = {
                "id": "taco_%d" % r["idx"],
                "qhash": r["qhash"],
                "split": r["split"],
                "source": r["source"],
                "difficulty": r["difficulty"],
                "question": r["question"],
                "url": r.get("url"),
                "tests": r["tests"],
                "n_tests": r["n_tests"],
                "gold": r["gold"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print()
    print("已写", out)


if __name__ == "__main__":
    main()
