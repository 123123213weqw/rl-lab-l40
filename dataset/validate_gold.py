"""阶段 1: 验证 TACO 的测试用例是否可信。

标准: 至少一个 Python 参考解能通过全部用例。通不过的题丢弃——
否则模型做对了也会被判错，污染整个数据集。

输出: validated.json  (题目 id、测试、gold 索引、难度、来源)

用法:
  python validate_gold.py --limit 200 --workers 32 --out results/val_probe.json
"""
import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CACHE = "/data/wangyue/rl-lab/hf-cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-try", type=int, default=3, help="每题最多试几个参考解")
    ap.add_argument("--timeout", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from taco_lib import load_taco, parse_io, parse_solutions, looks_python

    ds = load_taco(CACHE)
    print("TACO 总数:", len(ds), flush=True)

    cands = []
    skipped = Counter()
    for i, x in enumerate(ds):
        tests = parse_io(x.get("input_output"))
        if tests is None:
            skipped["no_tests"] += 1
            continue
        sols = [s for s in parse_solutions(x.get("solutions")) if looks_python(s)]
        if not sols:
            skipped["no_python_gold"] += 1
            continue
        cands.append({
            "idx": i,
            "tests": tests,
            "sols": sols,
            "difficulty": x.get("difficulty") or "UNKNOWN",
            "source": x.get("source") or "unknown",
            "question": (x.get("question") or "")[:4000],
            "url": x.get("url"),
            "time_limit": x.get("time_limit"),
        })
    print("可验证候选:", len(cands), "| 跳过:", dict(skipped), flush=True)

    if args.limit:
        cands = cands[:args.limit]

    tasks = [(c["idx"], c["tests"], c["sols"], args.max_try) for c in cands]
    by_idx = {c["idx"]: c for c in cands}

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_validate, t, args.timeout): t[0] for t in tasks}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            c = by_idx[r["id"]]
            r["difficulty"] = c["difficulty"]
            r["source"] = c["source"]
            results.append(r)
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                print("  %d/%d  %.1f/s" % (done, len(tasks), done / el), flush=True)

    el = time.time() - t0
    valid = [r for r in results if r["valid"]]
    print()
    print("验证完成: %d 题, 耗时 %.0fs (%.1f 题/s)" % (len(results), el, len(results) / el))
    print("测试可信: %d/%d = %.1f%%" % (len(valid), len(results), 100.0 * len(valid) / max(1, len(results))))

    print()
    print("可信率按来源:")
    src_all = Counter(r["source"] for r in results)
    src_ok = Counter(r["source"] for r in valid)
    for s, n in src_all.most_common():
        print("  %-16s %5d/%5d  %5.1f%%" % (s, src_ok[s], n, 100.0 * src_ok[s] / n))

    print()
    print("可信率按难度:")
    d_all = Counter(r["difficulty"] for r in results)
    d_ok = Counter(r["difficulty"] for r in valid)
    for d, n in d_all.most_common():
        print("  %-20s %5d/%5d  %5.1f%%" % (d, d_ok[d], n, 100.0 * d_ok[d] / n))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in valid:
        c = by_idx[r["id"]]
        payload.append({
            "idx": r["id"],
            "difficulty": c["difficulty"],
            "source": c["source"],
            "question": c["question"],
            "url": c["url"],
            "tests": c["tests"],
            "gold": c["sols"][r["gold_index"]],
            "n_tests": r["n_tests"],
        })
    out.write_text(json.dumps(payload, ensure_ascii=False))
    print()
    print("已写", out, "(%d 题)" % len(payload))

    stats = {"total": len(results), "valid": len(valid),
             "by_source": {s: [src_ok[s], src_all[s]] for s in src_all},
             "by_difficulty": {d: [d_ok[d], d_all[d]] for d in d_all},
             "skipped": dict(skipped),
             "seconds": round(el, 1)}
    Path(str(out).replace(".json", ".stats.json")).write_text(json.dumps(stats, indent=2))


def _validate(task, timeout):
    from taco_lib import validate_one
    pid, tests, sols, max_try = task
    return validate_one((pid, tests, sols, max_try))


if __name__ == "__main__":
    main()
