"""在沙箱内运行一个候选解法，跑完 tests.json 里所有用例。

不 import 解法，而是 exec + 重定向 stdio，这样竞赛风格的 stdin/stdout
程序和函数式程序都能跑。每次 exec 用全新 globals，避免状态泄漏。

输入文件（同目录）:
  sol.py     候选解法源码
  tests.json {"inputs": [...], "outputs": [...]}
输出（stdout）:
  JSON: {"passed": k, "total": n, "errors": [...], "first_fail": {...}}
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    """竞赛评测的常规归一化：整体 strip，逐行去行尾空白，去首尾空行。

    不做顺序无关比较——多解题目应被剔除而不是被放过，否则评分不可信。
    """
    if s is None:
        return ""
    txt = str(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


SENTINEL = "__RESULT__"


def emit(obj):
    """用哨兵前缀输出，避免被候选解泄漏到 stdout 的内容干扰解析。"""
    real = sys.__stdout__
    real.write(SENTINEL + " " + json.dumps(obj) + "\n")
    real.flush()


def main():
    with open(os.path.join(HERE, "tests.json")) as f:
        tests = json.load(f)
    with open(os.path.join(HERE, "sol.py")) as f:
        src = f.read()

    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])
    if len(inputs) != len(outputs):
        n = min(len(inputs), len(outputs))
        inputs, outputs = inputs[:n], outputs[:n]

    real_stdout = sys.stdout
    real_stdin = sys.stdin
    passed = 0
    errors = []
    first_fail = None

    try:
        code = compile(src, "sol.py", "exec")
    except SyntaxError as e:
        sys.stdout = real_stdout
        emit({"passed": 0, "total": len(inputs),
              "errors": ["SyntaxError: %s" % e.msg], "first_fail": None})
        return

    for i, (inp, exp) in enumerate(zip(inputs, outputs)):
        buf = io.StringIO()
        sys.stdin = io.StringIO(inp if isinstance(inp, str) else str(inp))
        sys.stdout = buf
        err = None
        try:
            exec(code, {"__name__": "__main__", "__builtins__": __builtins__})
        except SystemExit:
            pass
        except BaseException as e:  # noqa: BLE001 - 任何异常都算该用例失败
            err = "%s: %s" % (type(e).__name__, str(e)[:120])
        finally:
            got = buf.getvalue()
            sys.stdout = real_stdout
            sys.stdin = real_stdin

        want = exp if isinstance(exp, str) else str(exp)
        if norm(got) == norm(want):
            passed += 1
        else:
            if err:
                errors.append("case %d: %s" % (i, err))
            if first_fail is None:
                first_fail = {
                    "case": i,
                    "input": (inp if isinstance(inp, str) else str(inp))[:300],
                    "want": norm(want)[:300],
                    "got": norm(got)[:300],
                    "err": err,
                }

    emit({"passed": passed, "total": len(inputs),
          "errors": errors[:5], "first_fail": first_fail})


if __name__ == "__main__":
    main()
