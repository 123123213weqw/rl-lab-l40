"""阶段 4: 用 pass rate 过滤，生成训练集与评测基准。

设计决策:
  - 训练集: pass_rate ∈ [0.1, 0.9]，保证组内有方差（G=8 时零方差率最低）
  - 评测集: pass_rate ∈ [0.2, 0.8]，更窄的带，保证有 headroom 且不太难
  - 评测指标用「多次采样的平均 pass rate」而非二元 pass@1——
    前者方差低得多，同样的题量能检测更小的改进
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import argparse
import json
import math
from collections import Counter

POOL = ROOT / "data/taco_pool.jsonl"


def load_pool():
    return {r["id"]: r for r in
            (json.loads(l) for l in POOL.read_text().split("\n") if l.strip())}


def raw_p(r):
    """原始通过比例。用于判断饱和——必须保留 0 和 1 这两个端点。"""
    return r["correct"] / r["n"] if r["n"] else 0.0


def lap(r):
    """Laplace 平滑，仅用于保留题的统计估计，不用于过滤判断。"""
    return (r["correct"] + 1) / (r["n"] + 2)


def zero_var_prob(p, G):
    return p ** G + (1 - p) ** G


def mde(n, d=0.20, z=2.80):
    """配对设计、80% power 下能检测的最小提升。"""
    return z * math.sqrt(d / n) if n else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-passrate", default=str(ROOT / "results/train_passrate.json"))
    ap.add_argument("--eval-passrate", default=str(ROOT / "results/code_eval_passrate.json"))
    ap.add_argument("--train-lo", type=float, default=0.10)
    ap.add_argument("--train-hi", type=float, default=0.90)
    ap.add_argument("--eval-lo", type=float, default=0.125)
    ap.add_argument("--eval-hi", type=float, default=0.875)
    ap.add_argument("--outdir", default=str(ROOT / "data"))
    args = ap.parse_args()

    pool = load_pool()
    trp = Path(args.train_passrate)
    tr = json.loads(trp.read_text()) if trp.exists() else []
    ev = json.loads(Path(args.eval_passrate).read_text())
    print("train 已测:", len(tr), "| eval/dev 已测:", len(ev))

    def select(recs, lo, hi, must_split):
        out = []
        for r in recs:
            if r["idx"] not in pool:
                continue
            pr = raw_p(r)
            if lo <= pr <= hi and pool[r["idx"]]["split"] == must_split:
                item = dict(pool[r["idx"]])
                item["raw_p"] = pr
                item["pass_rate"] = round(lap(r), 4)
                item["n_samples"] = r["n"]
                out.append(item)
        return out

    # 训练集严格剔除全对/全错：这些题在 GRPO 里产生零优势
    train_all = select(tr, 0.0, 1.0, "train")
    train = [x for x in train_all if 0.0 < x["raw_p"] < 1.0]
    print("训练集: 候选 %d，剔除全对/全错后 %d 题"
          % (len(train_all), len(train)))

    # 评测集：用原始比例带 [eval_lo, eval_hi]，不额外剔除端点以外
    # 但保留 raw_p 在 (0,1) 开区间内的题，其余（全对/全错）无测量价值
    ev_sel = []
    for want in ("eval", "dev"):
        ev_sel += [x for x in select(ev, args.eval_lo, args.eval_hi, want)]

    for name, sel in (("训练集", train), ("评测基准", ev_sel)):
        print()
        print("=== %s ===" % name)
        print("题数:", len(sel))
        if not sel:
            continue
        G = 8
        zv = sum(zero_var_prob(r["pass_rate"], G) for r in sel) / len(sel)
        print("G=8 零方差率: %.1f%%  -> 有效组 %.1f%%" % (100 * zv, 100 * (1 - zv)))
        print("平均 pass rate: %.3f" % (sum(r["pass_rate"] for r in sel) / len(sel)))
        print("难度:", dict(Counter(r["difficulty"] for r in sel)))
        print("来源:", dict(Counter(r["source"] for r in sel)))
        print("MDE (d=0.20, 80%% power): %.1f%%" % (100 * mde(len(sel))))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 训练集：去掉 gold（训练时不该有答案），保留测试
    if not train:
        print()
        print("!! 训练集为空（train pass rate 还没测完），本次只产出评测基准")
    with (out / "train_code.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps({
                "id": r["id"], "question": r["question"], "tests": r["tests"],
                "n_tests": r["n_tests"], "difficulty": r["difficulty"],
                "source": r["source"], "pass_rate": r["pass_rate"],
            }, ensure_ascii=False) + "\n")

    # 评测集：保留 gold 以便复算 gold 通过率
    with (out / "eval_code.jsonl").open("w") as f:
        for r in ev_sel:
            f.write(json.dumps({
                "id": r["id"], "question": r["question"], "tests": r["tests"],
                "n_tests": r["n_tests"], "difficulty": r["difficulty"],
                "source": r["source"], "pass_rate": r["pass_rate"],
                "gold": r["gold"], "url": r.get("url"),
            }, ensure_ascii=False) + "\n")

    print()
    print("已写:")
    print(" ", out / "train_code.jsonl", len(train), "题")
    print(" ", out / "eval_code.jsonl", len(ev_sel), "题")

    stats = {
        "train": {"n": len(train),
                  "zero_var_G8": round(sum(zero_var_prob(r["pass_rate"], 8) for r in train) / max(1, len(train)), 4),
                  "by_difficulty": dict(Counter(r["difficulty"] for r in train)),
                  "by_source": dict(Counter(r["source"] for r in train))},
        "eval": {"n": len(ev_sel),
                 "zero_var_G8": round(sum(zero_var_prob(r["pass_rate"], 8) for r in ev_sel) / max(1, len(ev_sel)), 4),
                 "mde": round(mde(len(ev_sel)), 4),
                 "by_difficulty": dict(Counter(r["difficulty"] for r in ev_sel)),
                 "by_source": dict(Counter(r["source"] for r in ev_sel))},
        "thresholds": {"train": [args.train_lo, args.train_hi],
                       "eval": [args.eval_lo, args.eval_hi]},
    }
    (out / "code_dataset_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
