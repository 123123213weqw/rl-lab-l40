"""统一的路径解析。

所有脚本从这里拿数据目录，而不是用 ``Path(__file__).parent`` ——
因为脚本分布在不同子目录里，各自算自己的 parent 会指向错误位置。

导入本模块会把仓库根、sandbox/、train/、eval/ 加入 sys.path，
这样 ``import taco_lib`` / ``import code_reward`` 在任何子目录里都能工作。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
SANDBOX_DIR = ROOT / "sandbox"
TRAIN_DIR = ROOT / "train"
EVAL_DIR = ROOT / "eval"

for _p in (str(ROOT), str(SANDBOX_DIR), str(TRAIN_DIR), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
