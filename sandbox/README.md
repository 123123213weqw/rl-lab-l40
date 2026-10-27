# sandbox

执行模型生成的不可信代码。三个文件必须同目录，不要拆开。

| 文件 | 作用 |
|---|---|
| `sandbox_run.sh` | 用 `unshare` + `prlimit` 建立隔离环境并运行 |
| `driver.py` | 沙箱内的判分器，跑测试用例，用 `__RESULT__` 前缀输出 JSON |
| `taco_lib.py` | 加载数据、调度判分、解析结果。`run_in_sandbox()` 是入口 |

## 为什么不用 docker

机器上没有 docker 权限，也没有 nsjail/bwrap。用内核命名空间自建：

```
unshare -Urnm --fork --pid --mount-proc
  user / net / mount / pid 四个命名空间
  tmpfs 遮蔽 /home /data1 /data /myjfs /root /opt
prlimit: cpu / as / nproc / nofile / fsize / core
```

## 对抗性验证

| 攻击 | 结果 |
|---|---|
| 写 `/data1/…/PWNED` | FileNotFoundError，宿主无文件 |
| 连 `1.1.1.1:53` | OSError |
| fork 炸弹 | 4–12 个后 BlockingIOError |
| 死循环 | 4.03s 被 SIGKILL |
| 分配 4GB | 被 RLIMIT_AS 杀 |

## 两个坑

1. **`__RESULT__` 哨兵前缀**：候选代码会往 stdout 泄漏内容，
   不隔离的话会被当成结果解析而崩溃。
2. **`sys.set_int_max_str_digits(0)`**：TACO 有 >4300 位的整数字面量，
   触发 Python 的 DoS 防护，`json.loads` 直接抛 ValueError。

## 开销

正常代码 ~52 ms，死循环 ~3 s（受 `timeout` 限制）。
批量判分用进程池，48–64 并发。
