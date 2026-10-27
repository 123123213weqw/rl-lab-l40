"""两个 checkpoint 在逐题 dense 口径上的配对 bootstrap 检验。

同一批题、同一口径 -> 配对比较能消掉题目难度方差，比看总体均值灵敏得多。
"""
import json
import random
import statistics
import sys


def main(a_path, b_path, n_boot=10000):
    A = {x["id"]: x["dense"] for x in json.load(open(a_path))["per_problem"]}
    B = {x["id"]: x["dense"] for x in json.load(open(b_path))["per_problem"]}
    ids = sorted(set(A) & set(B))
    if not ids:
        sys.exit("没有共同题目")
    a = [A[i] for i in ids]
    b = [B[i] for i in ids]
    diffs = [y - x for x, y in zip(a, b)]
    obs = statistics.mean(diffs)

    rng = random.Random(311)
    boots = []
    n = len(ids)
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    p_pos = sum(1 for x in boots if x <= 0) / n_boot

    print("共同题目: %d" % n)
    print("A 均值 %.4f   B 均值 %.4f" % (statistics.mean(a), statistics.mean(b)))
    print("配对差 (B-A): %+.4f  [%.4f, %.4f]  95%% CI" % (obs, lo, hi))
    print("变好 %d / 变坏 %d / 不变 %d" % (
        sum(1 for d in diffs if d > 1e-9),
        sum(1 for d in diffs if d < -1e-9),
        sum(1 for d in diffs if abs(d) <= 1e-9)))
    print("bootstrap 中差值 <= 0 的比例: %.4f  (单侧 p)" % p_pos)
    print("显著 (95%% CI 不含 0): %s" % ("是" if lo > 0 or hi < 0 else "否"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
