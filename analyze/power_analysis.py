"""评测基准的统计功效分析：二元 pass@1 vs 多次采样 pass rate。

为什么用 pass rate 当指标:
  配对比较时题目本身的难度差异会抵消，剩下的方差来自采样噪声。
  每题采 n 次，噪声降到 1/n，等效于把题量放大 n 倍。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import json
import math


P = ROOT / "data" / "eval_code.jsonl"


def load():
    if not P.exists():
        return None
    return [json.loads(l) for l in P.read_text().split("\n") if l.strip()]


def mde_binary(N, d=0.20, z=2.80):
    """配对二元指标：不一致率 d，N 题。"""
    return z * math.sqrt(d / N)


def mde_passrate(N, n, d=0.20, z=2.80):
    """配对 pass-rate 指标：每题采 n 次。不一致率同样为 d，但被 n 次采样摊薄。"""
    return z * math.sqrt(d / (N * n))


def main():
    rows = load()
    if not rows:
        print("还没有 eval_code.jsonl，先跑 build_splits.py")
        return

    N = len(rows)
    print("评测集: %d 题" % N)
    print("平均 pass rate: %.3f" % (sum(r["pass_rate"] for r in rows) / N))
    print()

    print("同一批题，不同采样次数下的最小可检测提升 (MDE, 80%% power, d=0.20):")
    print("%-8s %-14s %-14s" % ("n/题", "二元 pass@1", "pass rate"))
    print("-" * 40)
    for n in (1, 4, 8, 16, 32):
        b = mde_binary(N)
        p = mde_passrate(N, n)
        print("%-8d %-14s %-14s" % (
            n,
            "%.1f%%" % (100 * b) if n == 1 else "(同一次采样)",
            "%.1f%%" % (100 * p),
        ))

    print()
    print("解读：")
    print("  - 用二元 pass@1，%d 题只能检测 %.1fpp 以上的提升" % (N, 100 * mde_binary(N)))
    print("  - 改用每题 8 次采样的 pass rate，降到 %.1fpp" % (100 * mde_passrate(N, 8)))
    print("  - 等价于把题量放大 8 倍，而推理成本只增 8 倍（本来就该多采样）")

    print()
    print("对比公开基准:")
    print("  MATH-500      n=500  MDE=5.6pp")
    print("  AIME 2024     n=30   MDE=22.9pp  (不可用)")
    print("  本基准        n=%d   MDE=%.1fpp (pass rate, 8 采样)" % (N, 100 * mde_passrate(N, 8)))


if __name__ == "__main__":
    main()
