"""veRL custom reward for competitive-programming tasks (TACO-derived).

与 GSM8K 那个严格二元奖励的区别
----------------------------------
原奖励：格式合规 + 答案完全正确才给 1，否则 0。在 93% 已饱和的 GSM8K 上，
这意味着组内样本几乎全是 1 或全是 0，GRPO 的优势项恒为零。

本奖励：在沙箱里跑测试，按**通过的测试比例**给分（dense）。一道题通过
15/20 也拿 0.75，而不是 0。这让「部分正确」产生非零优势。

评分口径（环境变量控制，便于做对照实验）
----------------------------------------
CODE_REWARD_MODE
  dense  (默认)  score = passed / total
  binary         score = 1.0 if passed == total else 0.0

安全
----
模型生成的代码一律在沙箱里执行（unshare + prlimit，见 sandbox_run.sh）。
reward 模块自身只负责抽取代码和调度，不 import 候选代码。

接口
----
compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
返回 dict：score 是训练用的标量，其余是日志指标。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS, DOCS  # noqa: E402

import json
import os
import re

# paths 模块已把 sandbox/ 加入 sys.path，这里直接导入即可
from taco_lib import run_in_sandbox  # noqa: E402

# ---- 可调参数 ----
MODE = os.environ.get("CODE_REWARD_MODE", "dense").lower()
TIMEOUT = int(os.environ.get("CODE_REWARD_TIMEOUT", "3"))
MEM_MB = int(os.environ.get("CODE_REWARD_MEM_MB", "512"))
MAX_TESTS = int(os.environ.get("CODE_REWARD_MAX_TESTS", "0"))       # 0 = 不截断
EXPECTED_SOURCE = os.environ.get("CODE_REWARD_SOURCE", "taco_code")

# 从模型输出中抽代码块
FENCE = re.compile(r"```(?:python3?|py)?[ \t]*\n(.*?)```", re.S)
# 兜底：没有围栏时，若整段能被 compile 就当代码
MAX_RESPONSE_CHARS = 200_000


def extract_code(text):
    """优先取最后一个围栏代码块。

    取最后一个而不是第一个：模型常先给思路再给最终代码。
    没有围栏时整段当作代码（下游 compile 会判掉非代码）。
    """
    if not isinstance(text, str) or not text or len(text) > MAX_RESPONSE_CHARS:
        return None
    blocks = FENCE.findall(text)
    if blocks:
        code = blocks[-1]
    else:
        code = text
    code = code.strip()
    return code or None


def parse_tests(ground_truth, extra_info):
    """测试用例可以从 ground_truth 或 extra_info 来。

    ground_truth 是 parquet 里的 reward_model.ground_truth 字符串，
    对代码任务存的是 {"tests": {...}} 的 JSON。
    """
    for src in (ground_truth, (extra_info or {}).get("tests")):
        if src is None:
            continue
        obj = src
        if isinstance(src, str):
            try:
                obj = json.loads(src)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        if isinstance(obj, dict) and "inputs" in obj and "outputs" in obj:
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("tests"), dict):
            return obj["tests"]
    return None


def _fail(reason):
    return {
        "score": 0.0, "acc": 0.0, "pass_ratio": 0.0,
        "tests_passed": 0, "tests_total": 0,
        "no_code": float(reason == "no_code"),
        "bad_tests": float(reason == "bad_tests"),
        "timeout": 0.0,
    }


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """veRL custom-reward callable。"""
    if data_source != EXPECTED_SOURCE:
        raise ValueError("Unexpected data source: %r (expected %r)"
                         % (data_source, EXPECTED_SOURCE))

    tests = parse_tests(ground_truth, extra_info)
    if tests is None:
        return _fail("bad_tests")

    code = extract_code(solution_str)
    if code is None:
        return _fail("no_code")

    if MAX_TESTS > 0:
        n = min(len(tests["inputs"]), len(tests["outputs"]), MAX_TESTS)
        tests = {"inputs": tests["inputs"][:n], "outputs": tests["outputs"][:n]}

    total = min(len(tests["inputs"]), len(tests["outputs"]))
    if total == 0:
        return _fail("bad_tests")

    result = run_in_sandbox(code, tests, timeout=TIMEOUT, mem_mb=MEM_MB)
    passed = int(result.get("passed", 0))
    # driver 返回的 total 可能因截断而不同，以实际为准
    real_total = int(result.get("total", total)) or total
    ratio = passed / real_total if real_total else 0.0
    is_all = passed == real_total > 0

    if MODE == "binary":
        score = 1.0 if is_all else 0.0
    else:
        score = ratio

    # 沙箱整体超时（进程被 SIGKILL，driver 没机会输出）时 result 里没有
    # __RESULT__，run_in_sandbox 会返回 errors=["no_result"]
    timed_out = "no_result" in (result.get("errors") or [])

    return {
        "score": float(score),
        "acc": float(is_all),
        "pass_ratio": float(ratio),
        "tests_passed": passed,
        "tests_total": real_total,
        "no_code": 0.0,
        "bad_tests": 0.0,
        "timeout": float(timed_out),
    }
