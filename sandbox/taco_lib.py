"""TACO 代码数据集的加载、Python 判定、沙箱评测。

设计要点:
  - 每个候选解法在独立沙箱目录中执行，不共享状态
  - 判定"测试可信"的唯一标准: 至少一个参考解能通过全部用例
  - 并行用进程池，每个任务自己建临时目录
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# TACO 里有超长整数字面量（>4300 位），会触发 Python 的 DoS 防护上限
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

warnings.filterwarnings("ignore", category=SyntaxWarning)

HERE = Path(__file__).resolve().parent
SANDBOX = HERE / "sandbox_run.sh"
DRIVER = HERE / "driver.py"

# C/C++/Java/Go 等语言的显著特征，提前排除，省一次编解码
NON_PY = (
    "#include", "using namespace", "int main", "public class", "public static",
    "import java", "func main", "package main", "std::", "printf(", "scanf(",
    "System.out", "vector<", "void main", "import Foundation", "def main() ->",
)


def looks_python(src):
    """先用便宜的特征筛，再用 compile() 做权威判定。"""
    if not src or len(src) < 10 or len(src) > 20000:
        return False
    low = src
    if any(tok in low for tok in NON_PY):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile(src, "<s>", "exec")
    except (SyntaxError, ValueError, OverflowError, MemoryError):
        return False
    # 竞赛题必须有某种 IO
    return ("input(" in src) or ("sys.stdin" in src) or ("open(" in src)


def parse_io(raw):
    """input_output 字段是 JSON 字符串 -> dict。"""
    if not raw:
        return None
    try:
        d = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    ins, outs = d.get("inputs"), d.get("outputs")
    if not isinstance(ins, list) or not isinstance(outs, list):
        return None
    n = min(len(ins), len(outs))
    if n == 0:
        return None
    ins, outs = ins[:n], outs[:n]
    # 单用例输入过大时跳过（避免超时/内存炸）
    if any(len(str(x)) > 20000 for x in list(ins) + list(outs)):
        return None
    return {"inputs": [str(x) for x in ins], "outputs": [str(x) for x in outs]}


def parse_solutions(raw):
    if not raw:
        return []
    try:
        s = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if isinstance(s, str):
        return [s]
    if isinstance(s, list):
        return [x for x in s if isinstance(x, str)]
    return []


def run_in_sandbox(code, tests, timeout=8, mem_mb=512):
    """执行一次候选解法，返回 dict。任何失败都返回 passed=0。"""
    work = tempfile.mkdtemp(prefix="taco.", dir="/tmp")
    try:
        shutil.copy(DRIVER, os.path.join(work, "driver.py"))
        with open(os.path.join(work, "sol.py"), "w") as f:
            f.write(code)
        with open(os.path.join(work, "tests.json"), "w") as f:
            json.dump(tests, f)
        p = subprocess.run(
            ["bash", str(SANDBOX), str(timeout), str(mem_mb), work],
            capture_output=True, text=True, timeout=timeout + 30,
        )
        tail = {"passed": 0, "total": len(tests["inputs"]),
                "errors": ["no_result"], "first_fail": None}
        for line in (p.stdout or "").split("\n"):
            if line.startswith("__RESULT__ "):
                tail = json.loads(line[len("__RESULT__ "):])
                break
        if not isinstance(tail, dict) or "passed" not in tail:
            return {"passed": 0, "total": len(tests["inputs"]),
                    "errors": ["bad_result"], "first_fail": None}
        return tail
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, ValueError) as e:
        return {"passed": 0, "total": len(tests["inputs"]), "errors": [type(e).__name__], "first_fail": None}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def validate_one(args):
    """一个题目: 依次尝试参考解，直到有一个全过。

    失败时保留"通过用例最多"的那次尝试的细节，便于诊断是格式问题还是真错。
    """
    pid, tests, sols, max_try = args
    best = None
    for k, code in enumerate(sols[:max_try]):
        r = run_in_sandbox(code, tests)
        if r["total"] and r["passed"] == r["total"]:
            return {"id": pid, "valid": True, "gold_index": k,
                    "n_tests": r["total"], "n_gold_tried": k + 1, "best": None}
        if best is None or r["passed"] > best["passed"]:
            best = dict(r, gold_index=k)
    return {"id": pid, "valid": False, "gold_index": None,
            "n_tests": len(tests["inputs"]),
            "n_gold_tried": min(len(sols), max_try), "best": best}


def load_taco(cache_dir):
    """读 parquet（TACO 的 loader 脚本已被 datasets 5.x 废弃）。"""
    from huggingface_hub import HfApi, hf_hub_download
    from datasets import load_dataset

    api = HfApi()
    info = api.dataset_info("BAAI/TACO")
    files = [s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".parquet")]
    paths = [hf_hub_download("BAAI/TACO", f, repo_type="dataset", cache_dir=cache_dir) for f in files]
    return load_dataset("parquet", data_files=paths, split="train")
