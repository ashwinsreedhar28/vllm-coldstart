"""Classify every turn of the session-gap runs and summarise. python3 analyze.py runs/*/"""
import json, sys, collections
from pathlib import Path
tot = collections.Counter(); agg = collections.defaultdict(list)
for d in sys.argv[1:]:
    f = Path(d) / "rows.jsonl"
    if not f.exists(): continue
    st = json.loads((Path(d) / "state.json").read_text()); idle = st["endpoint"]["idleTimeout"]
    prev = None; seen = set(); c = collections.Counter()
    for r in map(json.loads, f.read_text().splitlines()):
        o = r["out"] if isinstance(r["out"], dict) else {}
        if r["turn"] < 0 or r["status"] != "COMPLETED" or "boot_id" not in o: continue
        gap = None if prev is None else round(r["t_submit"] - prev["t_done"], 1)
        if prev is None: kind = "first"
        elif o["requests_this_boot"] == 1: kind = "cold_new_worker" if o["worker_id"] not in seen else "cold_same_worker"
        elif r["delay_s"] < 0.2: kind = "warm"
        elif o["boot_id"] == prev["out"]["boot_id"]: kind = "revive_same"
        else: kind = "revive_other"
        seen.add(o["worker_id"]); c[kind] += 1
        if kind != "first": tot[kind] += 1
        frac = (o.get("cached_tokens") or 0) / max(o.get("prompt_tokens") or 1, 1)
        agg[kind].append((r["delay_s"], r["exec_s"], o.get("ttft_s"), frac))
        r["kind"], r["real_gap"] = kind, gap; prev = r
    print(f"{st['arm']:14s} idle {idle:>2}s fs {int('KV' in json.dumps(st)) if False else ''} turns {sum(c.values())} {dict(c)} workers {len(seen)}")
print("\nall arms, turns after the first:", dict(tot))
med = lambda xs: sorted(xs)[len(xs) // 2]
for k, v in agg.items():
    print(f"{k:17s} n={len(v):2d} delay median {med([x[0] for x in v]):6.2f} (max {max(x[0] for x in v):.2f})  exec median {med([x[1] for x in v]):6.2f}  ttft median {med([x[2] for x in v]):.3f}  cached fraction median {med([x[3] for x in v]):.3f} min {min(x[3] for x in v):.3f}")
