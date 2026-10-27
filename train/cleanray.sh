#!/bin/bash
pkill -9 -f "main_pp[o]" 2>/dev/null
pkill -9 -f "ray[:][:]" 2>/dev/null
pkill -9 -f "gcs_serve[r]" 2>/dev/null
pkill -9 -f "rayle[t]" 2>/dev/null
pkill -9 -f "verl[.]trainer" 2>/dev/null
pkill -9 -f "WorkerDic[t]" 2>/dev/null
sleep 5
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " "); do
  [ -n "$p" ] && kill -9 "$p" 2>/dev/null
done
sleep 3
echo "ray=$(pgrep -cf "ray[:][:]" 2>/dev/null || echo 0) gcs=$(pgrep -cf gcs_serve[r] 2>/dev/null || echo 0) raylet=$(pgrep -cf rayle[t] 2>/dev/null || echo 0) main=$(pgrep -cf "main_pp[o]" 2>/dev/null || echo 0)"
