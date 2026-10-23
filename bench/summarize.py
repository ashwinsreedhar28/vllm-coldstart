"""Summarize phases.jsonl into a per-phase table (median, min, max) per run."""
import json, sys
from collections import defaultdict
from statistics import median

ORDER = ["import_vllm", "load_model", "determine_available_memory", "initialize_from_config",
         "compile_or_warm_up_model", "handler_engine_ready", "handler_first_token", "delay_s", "wall_s"]

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = defaultdict(list)
for r in rows:
    if "phases" in r:  # run_baseline.py output: one line per run
        for p in r["phases"]:
            by[p["phase"]].append(p["secs"])
        by["wall_s"].append(r["wall_s"]); by["delay_s"].append(r["delay_s"])
    else:              # raw phases.jsonl
        by[r["phase"]].append(r["secs"])
print(f"{'phase':30} {'n':>3} {'median':>8} {'min':>8} {'max':>8}")
for p in ORDER + sorted(set(by) - set(ORDER) - {"process_start"}):
    if p in by:
        v = by[p]
        print(f"{p:30} {len(v):3d} {median(v):8.2f} {min(v):8.2f} {max(v):8.2f}")
