"""合并 run_sharded.sh 产出的分片 pass rate 结果。

去重依据是题目 id：分片之间不应重叠，但重跑会产生重复。
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    shards = sorted((ROOT / "results/shards").glob("shard*.json"))
    if not shards:
        print("没有找到分片结果，先跑 run_sharded.sh")
        return

    all_recs = []
    for p in shards:
        recs = json.loads(p.read_text())
        print("  %s: %d 条" % (p.name, len(recs)))
        all_recs.extend(recs)

    seen, uniq = set(), []
    for r in all_recs:
        if r["idx"] not in seen:
            seen.add(r["idx"])
            uniq.append(r)
    print("去重后: %d（剔除重复 %d）" % (len(uniq), len(all_recs) - len(uniq)))

    out = ROOT / "results/train_passrate.json"
    out.write_text(json.dumps(uniq, indent=2))
    print("已写", out)

    n = uniq[0]["n"]
    print()
    print("整体 pass@1: %.4f" % (sum(r["correct"] for r in uniq)
                                 / sum(r["n"] for r in uniq)))
    hist = Counter(r["correct"] for r in uniq)
    print("每题 %d 采样，通过数分布:" % n)
    for k in range(n + 1):
        c = hist.get(k, 0)
        print("  %d/%d : %5d (%5.1f%%)" % (k, n, c, 100 * c / len(uniq)))
    ambi = sum(1 for r in uniq if 0 < r["correct"] < r["n"])
    print()
    print("有分化 (0<p<1): %d (%.1f%%)" % (ambi, 100 * ambi / len(uniq)))


if __name__ == "__main__":
    main()
