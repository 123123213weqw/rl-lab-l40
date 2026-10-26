"""从分好的池子里挑出指定 split，导出给 grade_candidates.py。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

pool = [json.loads(l) for l in ROOT / "data" / "taco_pool.jsonl".read_text().split("\n") if l.strip()]
want = set(sys.argv[1].split(","))
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
out = sys.argv[3]

sel = [r for r in pool if r["split"] in want]
if limit:
    sel = sel[:limit]
Path(out).write_text(json.dumps(sel, ensure_ascii=False))
print("selected %d from %s -> %s" % (len(sel), sorted(want), out))
