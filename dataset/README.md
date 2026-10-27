# dataset

TACO 数据流水线。按执行顺序：

| 脚本 | 作用 | 产物 |
|---|---|---|
| `validate_gold.py` | 验证测试用例可信（gold 解必须全过） | `results/taco_validated.json` |
| `build_code_dataset.py` | 剔除坏源、去重、哈希切分 | `data/taco_pool.jsonl` |
| `grade_candidates.py` | 基座采样 + 沙箱判分 | `results/train_passrate.json` |
| `build_splits.py` | 按 pass rate 过滤，切出训练集与评测 | `data/{train,eval}_code.jsonl` |
| `prepare_code_data.py` | 转成 veRL parquet | `data/verl/*.parquet` |
| `merge_shards.py` | 合并多卡分片的 pass rate | `results/train_passrate.json` |
| `pick_split.py` | 按 split 挑子集导出 | 临时 json |
| `dump_eval_sets.py` | 导出 ARC / GSM8K 用于能力对照 | `data/{arc,gsm8k_neutral}.jsonl` |
| `dump_math500.py` | 导出 MATH-500 / AIME 2024 | `data/{math500,aime2024}.jsonl` |

`dump_*.py` 需要 `GRPO_BASELINE` 环境变量指向 baseline 仓库。

## 关键结论

**TACO 测试数据只有 65.0% 可信**（9,419 / 14,498）。来源分化极大：

| 来源 | 可信率 |
|---|---|
| hackerrank | 93.0% |
| atcoder | 78.6% |
| codeforces | 78.0% |
| aizu | 68.9% |
| codechef | 56.2% |
| hackerearth | 22.2% |
| geeksforgeeks | 0.1% |

`geeksforgeeks` 980 题只过 1 题，整源剔除而不是逐条挑。

## 一个会静默失效的坑

过滤 pass rate **不能**用 Laplace 平滑值。n=4 时 `(k+1)/(n+2)` 只能产生
{0.167, 0.333, 0.5, 0.667, 0.833} 五个值，全部落在 [0.1, 0.9] 内，
过滤器一条都滤不掉、也不报错。

必须用原始比例 `k/n` 判断饱和。

## 产出

- 训练集 1,333 题：G=8 零方差 3.0%，平均 pass rate 0.468
- 评测基准 326 题：G=8 零方差 8.5%，MDE 2.4pp（8 采样）
- train/eval 题干交集与 id 交集均为 0
