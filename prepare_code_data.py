"""把过滤后的代码数据转成 veRL 训练用的 parquet。

veRL 的 schema（与 GSM8K baseline 一致）：
  data_source   str   本脚本用 "taco_code"，与 code_reward.py 的 EXPECTED_SOURCE 对应
  prompt        list   chat messages
  ability       str    "code"
  reward_model  dict   {"style": "rule", "ground_truth": <JSON 字符串>}
  extra_info    dict   {"id", "split", "tests", ...}

ground_truth 里存的是 {"tests": {"inputs": [...], "outputs": [...]}} 的 JSON 字符串。
不走 extra_info 是因为 veRL 的 extra_info 在不同版本间传递不稳定，
而 reward_model.ground_truth 是稳定契约。
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 与原 GSM8K baseline 相同的指令风格：要求只给代码
INSTRUCTION = (
    "You are an expert competitive programmer. Write a complete Python 3 program "
    "that reads from standard input and writes to standard output. "
    "Output only the code, no explanation."
)


def qid(question):
    return hashlib.sha256(" ".join(question.split()).encode("utf-8")).hexdigest()


def make_record(item, split):
    question = item["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Empty question for id %r" % item.get("id"))
    tests = item["tests"]
    if not tests.get("inputs") or not tests.get("outputs"):
        raise ValueError("Empty tests for id %r" % item["id"])
    return {
        "data_source": "taco_code",
        "prompt": [{"role": "user",
                    "content": INSTRUCTION + "\n\nProblem:\n" + question.strip()}],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({"tests": tests}, ensure_ascii=False),
        },
        "extra_info": {
            "id": qid(question),
            "split": split,
            "difficulty": item.get("difficulty"),
            "source": item.get("source"),
            "pass_rate": item.get("pass_rate"),
            "n_tests": item.get("n_tests"),
            "url": item.get("url"),
        },
    }


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().split("\n") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data/train_code.jsonl"))
    ap.add_argument("--eval", default=str(ROOT / "data/eval_code.jsonl"))
    ap.add_argument("--outdir", default=str(ROOT / "data/verl"))
    args = ap.parse_args()

    from datasets import Dataset

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for name, path in (("train", args.train), ("test", args.eval)):
        items = load_jsonl(path)
        recs, seen = [], set()
        dup = 0
        for it in items:
            r = make_record(it, name)
            k = r["extra_info"]["id"]
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            recs.append(r)
        ds = Dataset.from_list(recs)
        p = out / ("%s.parquet" % name)
        ds.to_parquet(str(p))
        print("%-6s %5d 条 -> %s  (去重 %d, %d 列)"
              % (name, len(recs), p, dup, len(ds.column_names)))

    # 校验：抽一条能 round-trip
    ds = Dataset.from_parquet(str(out / "train.parquet"))
    r = ds[0]
    gt = json.loads(r["reward_model"]["ground_truth"])
    print()
    print("校验: data_source=%s, ability=%s" % (r["data_source"], r["ability"]))
    print("      tests %d 个, 首个输入 %r"
          % (len(gt["tests"]["inputs"]), gt["tests"]["inputs"][0][:40]))
    print("      prompt 长度 %d 字符" % len(r["prompt"][0]["content"]))


if __name__ == "__main__":
    main()
