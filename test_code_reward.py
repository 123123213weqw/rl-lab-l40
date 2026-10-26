"""code_reward.compute_score 的单元测试。

覆盖：正常评分、无代码、坏 tests、语法错误、部分通过、超时、data_source 校验。
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SRC = "taco_code"
TESTS = {"inputs": ["1", "2", "3", "10"], "outputs": ["1", "4", "9", "100"]}
GT = json.dumps({"tests": TESTS})

fails = []


def check(name, got, want, tol=1e-9):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    print("  %-42s got=%-8s want=%-8s %s" % (name, got, want, "OK" if ok else "FAIL"))
    if not ok:
        fails.append(name)


def main():
    import code_reward as R

    print("=== 正常评分 ===")
    gold = "```python\nx = int(input())\nprint(x*x)\n```"
    r = R.compute_score(SRC, gold, GT)
    check("gold 全过 -> score 1.0", r["score"], 1.0)
    check("  acc", r["acc"], 1.0)
    check("  tests_passed", r["tests_passed"], 4)

    print()
    print("=== 部分通过（dense 的核心）===")
    # 只在 x==1 和 x==2 时给出正确结果
    partial = "```python\nx = int(input())\nprint(x*x if x <= 2 else -1)\n```"
    r = R.compute_score(SRC, partial, GT)
    check("2/4 -> score 0.5", r["score"], 0.5)
    check("  pass_ratio", r["pass_ratio"], 0.5)
    check("  acc 为 0（非全过）", r["acc"], 0.0)

    print()
    print("=== 零分情形 ===")
    r = R.compute_score(SRC, "```python\nprint(-1)\n```", GT)
    check("全错 -> 0", r["score"], 0.0)
    r = R.compute_score(SRC, "```python\ndef f(:\n  pass\n```", GT)
    check("语法错 -> 0", r["score"], 0.0)
    r = R.compute_score(SRC, "```python\nwhile True: pass\n```", GT)
    check("死循环 -> 0 且标记 timeout", r["timeout"], 1.0)

    print()
    print("=== 输入异常 ===")
    r = R.compute_score(SRC, "no code here at all", GT)
    check("无围栏且非代码 -> 0", r["score"], 0.0)
    r = R.compute_score(SRC, "", GT)
    check("空响应 -> 0", r["score"], 0.0)
    r = R.compute_score(SRC, gold, "not json")
    check("坏 ground_truth -> 0 且标记", r["bad_tests"], 1.0)
    r = R.compute_score(SRC, gold, json.dumps({"tests": {"inputs": [], "outputs": []}}))
    check("空 tests -> 0", r["score"], 0.0)

    print()
    print("=== data_source 校验（应抛异常）===")
    try:
        R.compute_score("gsm8k_baseline", gold, GT)
        print("  %-42s FAIL 未抛异常" % "错误 source")
        fails.append("data_source guard")
    except ValueError:
        print("  %-42s OK" % "错误 source 抛出 ValueError")

    print()
    print("=== binary 模式 ===")
    os.environ["CODE_REWARD_MODE"] = "binary"
    import importlib
    importlib.reload(R)
    r = R.compute_score(SRC, partial, GT)
    check("binary: 2/4 -> 0", r["score"], 0.0)
    r = R.compute_score(SRC, gold, GT)
    check("binary: 4/4 -> 1", r["score"], 1.0)
    os.environ.pop("CODE_REWARD_MODE")

    print()
    print("=== 代码抽取：取最后一个围栏 ===")
    two = ("```python\nprint('thinking')\n```\n\nfinal:\n"
           "```python\nx=int(input())\nprint(x*x)\n```")
    r = R.compute_score(SRC, two, GT)
    check("两个围栏取最后一个 -> 1.0", r["score"], 1.0)

    print()
    if fails:
        print("失败 %d 项: %s" % (len(fails), fails))
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
