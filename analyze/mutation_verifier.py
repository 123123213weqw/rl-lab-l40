"""量化验证器的漏放率：变异测试。

RLVR 的一切都建立在「测试用例能区分对错」这个假设上。
本脚本直接检验它：对已验证正确的题（gold 解能全过），
用 AST 变异造出语义不同的程序，看有多少个**仍然通过全部测试**。

- 存活的变异体 = 验证器漏放 → 模型可以拿奖励而没解决问题
- 被杀死的变异体 = 测试起作用了

分两个层次：
  survived   通过全部测试
  equivalent 变异后与 gold 在随机输入上输出一致（视为等价，不算漏放）

用法:
  python analyze/mutation_verifier.py --limit 200 --per-problem 8 \
      --out results/mutation.json
"""
import argparse
import ast
import copy
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT, DATA, RESULTS  # noqa: E402

sys.set_int_max_str_digits(0)

CMP_SWAP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}


def classify(node):
    """节点可变异类型。收集与定位必须用同一套判定 + 同一遍历（ast.walk）。"""
    kinds = []
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        kinds.append("int")
    if isinstance(node, ast.Compare) and any(type(o) in CMP_SWAP for o in node.ops):
        kinds.append("cmp")
    if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        kinds.append("binop")
    if isinstance(node, ast.BoolOp):
        kinds.append("boolop")
    if isinstance(node, (ast.If, ast.While)):
        kinds.append("negate")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in ("min", "max"):
        kinds.append("minmax")
    return kinds


def enumerate_sites(tree):
    """返回 [(kind, (kind, ordinal))]；DFS/BFS 顺序由 ast.walk 唯一决定。"""
    sites, seen = [], Counter()
    for node in ast.walk(tree):
        for k in classify(node):
            sites.append((k, (k, seen[k])))
            seen[k] += 1
    return sites


def _find(tree, info):
    """按 (kind, ordinal) 定位节点，遍历顺序与 enumerate_sites 一致。"""
    kind_tag, ordinal = info[0], info[1]
    count = 0
    for node in ast.walk(tree):
        if kind_tag in classify(node):
            if count == ordinal:
                return node
            count += 1
    return None


def mutate(tree, kind, info, rng):
    """在原树副本上应用一处变异；失败返回 None。"""
    t = copy.deepcopy(tree)
    node = _find(t, info)
    if node is None:
        return None
    try:
        if kind == "int":
            v = node.value
            node.value = rng.choice([v + 1, v - 1, 0, 1, v * 2])
            if node.value == v:
                return None
        elif kind == "cmp":
            for i, op in enumerate(node.ops):
                if type(op) in CMP_SWAP:
                    node.ops[i] = CMP_SWAP[type(op)]()
                    break
            else:
                return None
        elif kind == "binop":
            if isinstance(node.op, ast.Add):
                node.op = ast.Sub()
            elif isinstance(node.op, ast.Sub):
                node.op = ast.Add()
            elif isinstance(node.op, ast.Mult):
                node.op = ast.FloorDiv()
            elif isinstance(node.op, ast.FloorDiv):
                node.op = ast.Mult()
            else:
                node.op = ast.Add()
        elif kind == "boolop":
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif kind == "negate":
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        elif kind == "minmax":
            node.func.id = "max" if node.func.id == "min" else "min"
        else:
            return None
    except (AttributeError, IndexError, TypeError):
        return None
    return t


def make_mutants(gold, n, seed):
    """生成 n 个变异体，返回 [(kind, code), ...]。"""
    rng = random.Random(seed)
    try:
        tree = ast.parse(gold)
    except SyntaxError:
        return []
    indexed = enumerate_sites(tree)
    if not indexed:
        return []

    rng.shuffle(indexed)
    out = []
    for kind, info in indexed[: n * 4]:
        if len(out) >= n:
            break
        mt = mutate(tree, kind, info, rng)
        if mt is None:
            continue
        try:
            code = ast.unparse(mt)
        except (AttributeError, ValueError):
            continue
        if code.strip() == gold.strip():
            continue
        out.append((kind, code))
    return out[:n]



def _randnum(digits, rng):
    """同长度随机数字串，首位非 0。"""
    if len(digits) == 1:
        return str(rng.randint(1, 9))
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(len(digits) - 1))


def perturb(inp, rng):
    """单点小扰动：只改一个数字 ±1/±2，尽量不违反输入约束。

    原来的做法是全部数字替换成随机同长度数，结果大量扰动违反约束
    （如 n 超范围）让 gold 自己也崩溃，差分验证无法进行。
    """
    if not isinstance(inp, str):
        return None
    spans = [m.span() for m in re.finditer(r"\d+", inp)]
    if not spans:
        return None
    a, b = rng.choice(spans)
    digits = inp[a:b]
    if len(digits) > 9:           # 太大的数不动，容易溢出/超时
        return None
    v = int(digits)
    delta = rng.choice([-2, -1, 1, 2])
    nv = v + delta
    if v >= 1 and nv < 1:         # 保持正整数约束
        nv = 1
    if nv == v:
        return None
    return inp[:a] + str(nv) + inp[b:]


def probe_gold(job):
    """用哨兵期望值跑 gold，从 first_fail.got 取到它的真实输出。"""
    from taco_lib import run_in_sandbox
    pid, gold, inp, timeout = job
    r = run_in_sandbox(gold, {"inputs": [inp], "outputs": ["__PROBE__"]}, timeout=timeout)
    ff = r.get("first_fail")
    if not ff or ff.get("err"):
        return pid, inp, None
    return pid, inp, ff.get("got")


def compare_mutant(job):
    """变异体在给定输入上是否与 gold 输出一致。"""
    from taco_lib import run_in_sandbox
    pid, kind, code, inp, want, timeout = job
    r = run_in_sandbox(code, {"inputs": [inp], "outputs": [want]}, timeout=timeout)
    return pid, kind, int(r.get("passed", 0)) == 1


def run_one(job):
    from taco_lib import run_in_sandbox
    jidx, pid, kind, code, tests, timeout = job
    try:
        r = run_in_sandbox(code, tests, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return jidx, pid, kind, "error", 0, type(e).__name__
    passed, total = int(r.get("passed", 0)), int(r.get("total", 0))
    if total == 0:
        return jidx, pid, kind, "error", passed, "no_tests"
    if passed == total:
        return jidx, pid, kind, "survived", passed, ""
    return jidx, pid, kind, "killed", passed, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated", default=str(RESULTS / "taco_validated.json"))
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--per-problem", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=3)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=311)
    ap.add_argument("--verify", type=int, default=0,
                    help="对每个存活变异体做几次扰动输入差分测试（0=不做）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.loads(Path(args.validated).read_text())
    print("已验证题数:", len(rows), flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.limit]
    print("抽样:", len(rows), "题, 每题最多", args.per_problem, "个变异体", flush=True)

    jobs = []
    n_mut = {}
    for r in rows:
        muts = make_mutants(r["gold"], args.per_problem, (r["idx"] * 2654435761) & 0xFFFF)
        n_mut[r["idx"]] = len(muts)
        for kind, code in muts:
            jobs.append((len(jobs), r["idx"], kind, code, r["tests"], args.timeout))

    print("变异体总数:", len(jobs), flush=True)
    if not jobs:
        print("没有可用的变异体（gold 太简单或解析失败）")
        return

    by_problem = defaultdict(list)
    by_kind = defaultdict(lambda: Counter())
    survivors = []          # (pid, kind, code)
    t0 = __import__("time").time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for f in as_completed(futs):
            jidx, pid, kind, status, passed, err = f.result()
            by_problem[pid].append(status)
            by_kind[kind][status] += 1
            if status == "survived":
                survivors.append((pid, kind, jobs[jidx][3]))
            done += 1
            if done % 500 == 0:
                el = __import__("time").time() - t0
                print("  %d/%d  %.1f/s" % (done, len(jobs), done / el), flush=True)

    el = __import__("time").time() - t0
    all_status = [s for v in by_problem.values() for s in v]
    c = Counter(all_status)
    tot = len(all_status)

    print()
    print("=" * 60)
    print("耗时 %.0fs  (%.1f 变异体/s)" % (el, tot / max(el, 1e-9)))
    print("变异体总数 %d  涉及 %d 题" % (tot, len(by_problem)))
    print()
    print("  survived  %6d  %5.1f%%   <- 验证器漏放" % (c["survived"], 100.0 * c["survived"] / tot))
    print("  killed    %6d  %5.1f%%" % (c["killed"], 100.0 * c["killed"] / tot))
    print("  error     %6d  %5.1f%%" % (c["error"], 100.0 * c["error"] / tot))
    print()
    print("每题至少一个变异体存活的题占比: %.1f%%" % (
        100.0 * sum(1 for v in by_problem.values() if "survived" in v) / len(by_problem)))
    print()
    print("按变异类型:")
    for k in sorted(by_kind):
        kc = by_kind[k]
        n = sum(kc.values())
        print("  %-8s n=%-5d survived %5.1f%%" % (k, n, 100.0 * kc["survived"] / max(n, 1)))
    print("=" * 60)

    # 按题统计，供后续与 pass_rate / 难度关联
    per_problem = []
    for pid, sts in by_problem.items():
        per_problem.append({
            "idx": pid,
            "n": len(sts),
            "survived": sts.count("survived"),
            "survive_rate": sts.count("survived") / len(sts),
        })
    rates = [p["survive_rate"] for p in per_problem]
    print()
    print("每题存活率: 均值 %.3f  中位 %.3f  p90 %.3f" % (
        statistics.mean(rates), statistics.median(rates),
        sorted(rates)[int(0.9 * len(rates))]))

    # ---------- 差分验证：幸存者是真的行为不同，还是语义等价 ----------
    equiv = None
    if args.verify and survivors:
        idx_gold = {r["idx"]: r["gold"] for r in rows}
        idx_tests = {r["idx"]: r["tests"] for r in rows}
        rng2 = random.Random(args.seed + 1)

        probe_map, jobs_g = {}, []
        seen = set()
        for pid, kind, code in survivors:
            ins = list(idx_tests[pid]["inputs"][:8])
            rng2.shuffle(ins)
            tries = 0
            for inp in ins * 4:                     # 多生成候选，后面按可用性截断
                if tries >= args.verify * 4:
                    break
                pin = perturb(inp, rng2)
                tries += 1
                if not pin or (pid, pin) in seen:
                    continue
                seen.add((pid, pin))
                probe_map.setdefault(pid, []).append(pin)
                jobs_g.append((pid, idx_gold[pid], pin, args.timeout))

        print()
        print("差分验证: %d 个幸存体, %d 组扰动输入" % (len(survivors), len(jobs_g)), flush=True)
        gold_out = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for pid, inp, out in ex.map(probe_gold, jobs_g):
                if out is not None:
                    gold_out[(pid, inp)] = out

        jobs_c = []
        for si, (pid, kind, code) in enumerate(survivors):
            usable = [i for i in probe_map.get(pid, []) if (pid, i) in gold_out]
            for inp in usable[: args.verify]:
                jobs_c.append((si, kind, code, inp, gold_out[(pid, inp)], args.timeout))

        diff, same = Counter(), Counter()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for si, kind, ok in ex.map(compare_mutant, jobs_c):
                (same if ok else diff)[si] += 1

        non_equiv_si = [si for si in range(len(survivors)) if diff[si] > 0]
        non_equiv = len(non_equiv_si)
        affected = {survivors[si][0] for si in non_equiv_si}
        equiv_n = sum(1 for si in range(len(survivors)) if diff[si] == 0 and same[si] > 0)
        unver = len(survivors) - non_equiv - equiv_n
        checked = non_equiv + equiv_n

        print("  行为不同（真漏放）: %d" % non_equiv)
        print("  输出一致（疑似等价）: %d" % equiv_n)
        print("  无法判定: %d" % unver)
        if checked:
            print()
            print("  幸存者中真漏放的比例: %.1f%%  (%d/%d)" % (
                100.0 * non_equiv / checked, non_equiv, checked))
            print("  全部变异体中的真漏放率: %.1f%%" % (100.0 * non_equiv / tot))
            print("  至少一个真漏放的题: %d/%d = %.1f%%" % (
                len(affected), len(by_problem),
                100.0 * len(affected) / len(by_problem)))
        equiv = {"survivors": len(survivors), "non_equivalent": non_equiv,
                 "equivalent_like": equiv_n, "unverified": unver,
                 "probes": len(jobs_c),
                 "problems_affected": len(affected),
                 "problems_total": len(by_problem),
                 "affected_idx": sorted(affected)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_problems": len(by_problem), "n_mutants": tot,
        "survived": c["survived"], "killed": c["killed"], "error": c["error"],
        "survive_rate": c["survived"] / tot,
        "problem_any_survive_rate": sum(1 for v in by_problem.values() if "survived" in v) / len(by_problem),
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "equivalence": equiv,
        "per_problem": per_problem,
    }, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
