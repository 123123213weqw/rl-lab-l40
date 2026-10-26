#!/bin/bash
# 在隔离环境中执行一个测试套件目录。
#
# 隔离：user+net+mount+pid namespace，tmpfs 遮蔽敏感目录，prlimit 限制资源。
# 用法: sandbox_run.sh <timeout_s> <mem_mb> <workdir>
#   workdir 内必须有 driver.py，以及它需要的其它文件。
#   driver.py 的 stdout 会被原样返回。

set -u
T="${1:?timeout}"
M="${2:?mem_mb}"
SRC="${3:?workdir}"

WORK=$(mktemp -d /tmp/sbx.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
cp -r "$SRC"/. "$WORK"/ || exit 99

export SANDBOX_WORK="$WORK"
export SANDBOX_TIMEOUT="$T"
export SANDBOX_MEM="$(( M * 1024 * 1024 ))"

unshare -Urnm --fork --pid --mount-proc bash -c '
  for d in /home /data1 /data /myjfs /root /opt; do
    mount -t tmpfs -o size=1m none "$d" 2>/dev/null
  done
  cd "$SANDBOX_WORK" || exit 98
  export HOME="$SANDBOX_WORK" TMPDIR="$SANDBOX_WORK" \
         PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
         PATH=/usr/bin:/bin
  exec prlimit \
      --cpu="$SANDBOX_TIMEOUT" \
      --as="$SANDBOX_MEM" \
      --nproc=32 \
      --nofile=256 \
      --fsize=16777216 \
      --core=0 \
      timeout -s KILL "$SANDBOX_TIMEOUT" \
      python3 -I driver.py
' 2>/dev/null

exit $?
