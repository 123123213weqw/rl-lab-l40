# TACO 代码数据集：构建记录

日期：2026-10-26。目录：`$RL_LAB/`，数据在 `data/`，结果在 `results/`。

## 为什么要自建

前序结论：GSM8K 对 Qwen3-8B 已饱和（86% 的题 8 次全对），
GRPO 的增益 98.8% 可由一个正则替代。需要换到有可学信号的任务。

代码任务的价值：单测可验证、几乎无法用格式 hack、信号比二元对错更细。

## 数据来源：BAAI/TACO

TACO 的 `datasets` loader 脚本在新版已废弃，**必须直接读 parquet**：

```python
from huggingface_hub import HfApi, hf_hub_download
files = [s.rfilename for s in HfApi().dataset_info("BAAI/TACO").siblings
         if s.rfilename.endswith(".parquet")]
paths = [hf_hub_download("BAAI/TACO", f, repo_type="dataset") for f in files]
ds = load_dataset("parquet", data_files=paths, split="train")
```

规模：**26,443 题**，全部带 `input_output` 测试用例，20,493 题有参考解。

## 关键一步：验证测试用例是否可信

TACO 的测试数据大量损坏。**判定标准：至少一个 Python 参考解能通过全部用例。**
不满足的题必须丢弃——否则模型做对了也会被判错，直接污染训练信号。

```
可验证候选        14,498   （另有 2,131 无测试、9,814 无 Python 参考解）
测试可信           9,419   = 65.0%
耗时               285s    @ 50.8 题/s
```

可信率按来源（**分化极大**）：

| 来源 | 可信/总数 | 可信率 |
|---|---|---|
| hackerrank | 745/801 | **93.0%** |
| atcoder | 950/1,208 | 78.6% |
| codeforces | 4,793/6,146 | 78.0% |
| aizu | 963/1,397 | 68.9% |
| codechef | 1,797/3,195 | 56.2% |
| hackerearth | 170/765 | 22.2% |
| geeksforgeeks | 1/980 | **0.1%** |
| leetcode | 0/5 | **0%** |

`geeksforgeeks` 980 题只通过 1 题，其测试数据在 TACO 中基本是坏的。
`leetcode`/`kattis`/`hackerearth` 同样低。**这些来源整体剔除**，
不是逐题挑选——保留它们只会引入噪声。

## 最终数据集

剔除坏源 + 题干哈希去重后：

```
9,169 题
  train  7,784
  dev      457
  eval     928
```

难度分布：EASY 3,181 / MEDIUM_HARD 1,585 / UNKNOWN 1,375 / HARD 1,279 /
MEDIUM 1,163 / VERY_HARD 586

来源分布：codeforces 4,714 / codechef 1,797 / aizu 963 / atcoder 950 / hackerrank 745

切分用**题干哈希的确定性切分**（非随机种子），保证：
- 可复现
- train/dev/eval 天然不相交
- 同一题干的不同版本会落在同一 split（哈希前先去重）

## 沙箱：执行不可信代码

模型生成的代码必须隔离执行。这台机器上没有 docker 权限、没有 nsjail/bwrap，
用 `unshare` + `prlimit` 实现：

```
unshare -Urnm --fork --pid --mount-proc
  ├── user ns   : 映射为 root，但对宿主无任何特权
  ├── net ns    : 无网络
  ├── mount ns  : tmpfs 遮蔽 /home /data1 /data /myjfs /root /opt
  └── pid ns    : 独立进程表
prlimit: cpu / as / nproc / nofile / fsize / core
```

### 对抗性验证（全部通过）

| 攻击 | 结果 |
|---|---|
| 写 `/data1/PWNED` | `FileNotFoundError`，宿主无文件产生 |
| 连 `1.1.1.1:53` | `OSError`，网络不可达 |
| fork 炸弹 | 每次 fork 在 4–12 个后 `BlockingIOError`（RLIMIT_NPROC） |
| 死循环 | 4.03s 被 SIGKILL |
| 分配 4GB | 被 `RLIMIT_AS` 杀 |
| 正常程序 `3 4` | 输出 `7`，exit 0 |

## 评分器设计

`driver.py` 在沙箱内运行，不 import 候选代码，而是 `exec` + 重定向 stdio，
这样竞赛风格的 stdin/stdout 程序和函数式程序都能跑。每次 exec 用全新 globals。

输出归一化：整体 strip、逐行去行尾空白、去首尾空行。
**不做顺序无关比较**——多解题目应被剔除，不应被放过。

结果用 `__RESULT__` 哨兵前缀输出：候选代码可能往 stdout 泄漏内容，
之前的版本因此把泄漏的数字当成了结果而崩溃。

## 已知的坑

1. **超长整数字面量**：TACO 有 >4300 位的整数，触发 Python 的
   `int_max_str_digits` 防护，`json.loads` 直接抛 `ValueError`。
   解法：`sys.set_int_max_str_digits(0)`。
2. **`compile()` 的 SyntaxWarning 污染输出**：用 `warnings.catch_warnings()` 抑制。
3. **部分题目的 gold 解与用例不匹配**：即使 5 个参考解也全不过，这类题被丢弃。
4. **多解题**：输出顺序不同导致 gold 解也过不了。这类题同样被丢弃——
   保留它们会让模型做对却被判错。

## 实测 pass rate（基座模型 Qwen3-8B，80 题 × 4 采样）

```
整体 pass@1: 22.19%  (71/320)

0/4 : 54 个 (67.5%)   ← 太难，无信号
1/4 :  7 个 ( 8.8%)
2/4 :  4 个 ( 5.0%)
3/4 :  4 个 ( 5.0%)
4/4 : 11 个 (13.8%)   ← 太简单，无信号
```

**重要：这否证了"代码天然落在可学带"的猜测。**
池子混合了难度，67.5% 的题模型完全做不出，零方差比例 81.3%。

但这正好说明**为什么要有 9,169 题**：过滤到可学带后仍有两千多题可用。

对比 GSM8K 的失败模式：

| | GSM8K | TACO（未过滤） |
|---|---|---|
| 整体 pass@1 | 93.4% | 22.2% |
| 全对 | 86.0% | 13.8% |
| 全错 | 2.4% | **67.5%** |
| 可学带 | 11.6% | 18.8% |
| 失败模式 | **太简单** | **太难** |

两头都错，但方向相反。代码集合有真实 headroom（22% → 有大量提升空间），
代价是需要难度过滤。

## 复现

```bash
cd $RL_LAB
export HF_ENDPOINT=https://hf-mirror.com

# 1. 验证测试可信（约 5 分钟）
./.venv-llm/bin/python validate_gold.py --max-try 5 --workers 48 \
    --out results/taco_validated.json

# 2. 构建数据集
./.venv-llm/bin/python build_code_dataset.py --out data/taco_pool.jsonl

# 3. 测 pass rate（需要 GPU）
PATH=$RL_LAB/.tools/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_VISIBLE_DEVICES=0 \
$VLLM_PY grade_candidates.py \
    --validated /tmp/evalset.json --model <ckpt> --n 8 --graders 40 \
    --out results/code_eval_passrate.json
```

## 产物

| 文件 | 内容 |
|---|---|
| `results/taco_validated.json` | 9,419 题（含测试、gold、难度、来源） |
| `results/taco_validated.stats.json` | 验证统计（按来源/难度） |
| `data/taco_pool.jsonl` | 9,169 题，已去重、已切分 |
| `results/code_probe.json` | 80 题 × 4 采样的 pass rate |
| `results/code_eval_passrate.json` | 1,385 题 × 8 采样的 pass rate |

## 下一步

1. 用 pass rate 过滤出 `p ∈ [0.1, 0.9]` 的子集，作为训练集（预期约 2,000–2,700 题）
2. eval split 用 pass rate 分层抽样，报 bootstrap CI
3. 奖励改为「通过测试数 / 总测试数」而非二元对错——这是代码任务相对数学的核心优势，
   信号更细，且不存在格式 hack 空间

## 局限

- 难度的度量依赖当前基座模型；模型变强后原可学带会向难端移动
- 测试用例已被截断到最多 20 个/题（控制单题评测耗时）
- TACO 的 Python 参考解只覆盖 20,493/26,443 题；其余题因无法验证测试可信而丢弃
- 不存在"多解答案"的完整处理；这类题被丢弃而非被正确评分
