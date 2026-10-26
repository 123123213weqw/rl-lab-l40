#!/bin/bash
# 把 train pass-rate 测量分片到 8 张卡并行。
# 每个分片: 独立 vLLM 实例（TP=1）+ 独立的沙箱评分进程池。
set -u
cd "$(dirname "$0")"

NSHARD=${1:-8}
N=${2:-4}
GRADERS=${3:-12}          # 每分片评分并发；8 x 12 = 96 < 128 线程
OUTDIR=results/shards
mkdir -p "$OUTDIR"

./.venv-llm/bin/python - "$NSHARD" <<'PY'
import json, sys
from pathlib import Path

n = int(sys.argv[1])
rows = json.loads(Path("/tmp/trainset.json").read_text())
per = (len(rows) + n - 1) // n
for i in range(n):
    chunk = rows[i * per:(i + 1) * per]
    if not chunk:
        continue
    p = Path("/tmp/train_shard%d.json" % i)
    p.write_text(json.dumps(chunk))
    print("shard %d: %d 题 -> %s" % (i, len(chunk), p))
PY

echo "=== 启动 $NSHARD 个分片 ==="
for i in $(seq 0 $((NSHARD - 1))); do
  if [ ! -f "/tmp/train_shard$i.json" ]; then continue; fi
  setsid nohup env \
    PATH=$PWD/.tools/bin:$PATH \
    CUDA_VISIBLE_DEVICES=$i \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    ${VLLM_PY:?set VLLM_PY to the vllm venv python} -u grade_candidates.py \
      --validated /tmp/train_shard$i.json \
      --model ${GRPO_BASELINE:?set GRPO_BASELINE}/execution/20261017/model \
      --n "$N" --limit 0 --graders "$GRADERS" \
      --out "$OUTDIR/shard$i.json" \
    > "$OUTDIR/shard$i.log" 2>&1 < /dev/null &
  disown
  echo "  shard $i -> GPU $i"
  sleep 3
done

echo "已全部提交"
