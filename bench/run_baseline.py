"""Step 2 runner: N cold starts of the vllm-coldstart baseline image on Runpod Serverless.

    python3 bench/run_baseline.py --env-file ~/.runpod.env --list-volumes
    python3 bench/run_baseline.py --env-file ~/.runpod.env --image ghcr.io/ashwinsreedhar28/vllm-coldstart:bench-v0.32.0 \
        --volume-id <id> --config A --runs 3

Per run: scale to 1 worker, send one job, wait, record the phases the handler returns,
scale to 0 and wait for the worker to disappear so the next run is cold. Always leaves
workersMax=0. Results go to bench/results/<config>-<timestamp>.jsonl (gitignored).

Configs (see bench/baseline.md): A default loader, B runai_streamer, C startup plan.
Stdlib only; API shapes follow day0-ready/pipeline/runpod_api.py (same author).
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

REST = "https://rest.runpod.io/v1"
JOBS = "https://api.runpod.ai/v2"
UA = "vllm-coldstart-bench/0.1 (+https://github.com/ashwinsreedhar28/vllm-coldstart)"
GPU_4090 = ["NVIDIA GeForce RTX 4090"]

CONFIGS = {
    "A": {},
    "B": {"LOAD_FORMAT": "runai_streamer", "LOADER_EXTRA": json.dumps({"concurrency": 32})},
    "C": {"VLLM_ENABLE_STARTUP_PLAN": "1"},
    # Runpod SDK fitness checks initialize CUDA in the handler process, which forces
    # vLLM to spawn (and re-import in) the EngineCore. Skip them so vLLM can fork.
    "D": {"RUNPOD_SKIP_AUTO_SYSTEM_CHECKS": "true"},
}
COMMON_ENV = {
    "MODEL_NAME": "Qwen/Qwen3-8B",
    "HF_HOME": "/runpod-volume/hf",
    "VLLM_CACHE_ROOT": "/runpod-volume/vllm-cache",
    "VLLM_COLDSTART_PHASES_OUT": "/runpod-volume/vllm-coldstart/phases.jsonl",
    "MAX_MODEL_LEN": "8192",
}


def load_key(env_file):
    if env_file:
        for line in Path(env_file).expanduser().read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("RUNPOD_API_KEY="):
                os.environ["RUNPOD_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    return key


def call(method, url, key, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Accept": "application/json", "User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:500]
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, str(e)


def ok(st): return 200 <= st < 300


def log(msg): print(time.strftime("%H:%M:%S"), msg, flush=True)


def workers(key, eid):
    st, h = call("GET", f"{JOBS}/{eid}/health", key)
    return {k: v for k, v in ((h or {}).get("workers") or {}).items() if v} if ok(st) and isinstance(h, dict) else {}


def wait_cold(key, eid, timeout=600, settle=3):
    """True once health reports no workers for `settle` consecutive polls (10 s apart)."""
    t, quiet = time.time(), 0
    while time.time() - t < timeout:
        quiet = quiet + 1 if not workers(key, eid) else 0
        if quiet >= settle:
            return True
        time.sleep(10)
    return False


def set_max_workers(key, eid, n, timeout=60):
    """PATCH workersMax and poll until the endpoint reports it (the API is eventually consistent)."""
    call("PATCH", f"{REST}/endpoints/{eid}", key, {"workersMax": n})
    t = time.time()
    while time.time() - t < timeout:
        st, e = call("GET", f"{REST}/endpoints/{eid}", key)
        if ok(st) and isinstance(e, dict) and e.get("workersMax") == n:
            return True
        time.sleep(3)
    return False


def one_run(key, eid, run_id, out, timeout=900):
    if not set_max_workers(key, eid, 1):
        log(f"run {run_id}: workersMax=1 did not apply"); return False
    t0 = time.time()
    for attempt in range(20):  # the jobs API sees the new workersMax a few seconds after REST does
        t0 = time.time()
        st, job = call("POST", f"{JOBS}/{eid}/run", key, {"input": {"prompt": "Explain paged attention in one paragraph.", "max_tokens": 32}})
        if ok(st) or "ENDPOINT_PAUSED" not in str(job):
            break
        time.sleep(3)
    if not ok(st):
        log(f"run {run_id}: submit failed {st} {job}"); return False
    jid = job["id"]
    status = None
    while time.time() - t0 < timeout:
        time.sleep(10)
        st, s = call("GET", f"{JOBS}/{eid}/status/{jid}", key)
        if ok(st) and s.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            status = s["status"]; break
    if status != "COMPLETED":
        log(f"run {run_id}: {status or 'DEADLINE'}; see https://console.runpod.io/serverless/user/endpoint/{eid}")
        call("POST", f"{JOBS}/{eid}/cancel/{jid}", key)
        return False
    o = s.get("output") or {}
    rec = {"run": run_id, "wall_s": round(time.time() - t0, 1), "delay_s": round((s.get("delayTime") or 0) / 1000, 1),
           "exec_s": round((s.get("executionTime") or 0) / 1000, 1), "ttft_s": o.get("ttft_s"), "gpu": o.get("gpu"), "worker_id": o.get("worker_id"), "phases": o.get("phases", [])}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"run {run_id} [{rec['gpu']}]: wall {rec['wall_s']}s delay {rec['delay_s']}s exec {rec['exec_s']}s ttft {rec['ttft_s']}s; "
        + ", ".join(f"{p['phase']}={p['secs']}" for p in rec["phases"] if p.get("secs")))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file")
    ap.add_argument("--list-volumes", action="store_true")
    ap.add_argument("--image", help="full image ref; prefer the bench-<sha> tag so a stale pull cache cannot serve an old build")
    ap.add_argument("--volume-id")
    ap.add_argument("--config", choices=CONFIGS, default="A")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="cold starts to run and discard first (weights download, compile cache fill)")
    ap.add_argument("--disk-gb", type=int, default=40)
    ap.add_argument("--keep", action="store_true", help="leave endpoint/template in place (scaled to 0)")
    ap.add_argument("--endpoint-id", help="reuse an existing endpoint instead of creating one (implies --keep, --warmup 0)")
    a = ap.parse_args()
    key = load_key(a.env_file)

    if a.list_volumes:
        st, v = call("GET", f"{REST}/networkvolumes", key)
        print(json.dumps(v, indent=1)[:3000]); return
    if not a.endpoint_id and not (a.image and a.volume_id):
        sys.exit("--image and --volume-id required (or --endpoint-id)")

    env = {**COMMON_ENV, **CONFIGS[a.config]}
    if a.endpoint_id:
        eid, tpl, a.keep, a.warmup = a.endpoint_id, None, True, 0
        set_max_workers(key, eid, 0)
        log(f"reusing endpoint {eid}; waiting for it to go cold")
        wait_cold(key, eid)
    else:
        name = f"vllm-coldstart-{a.config}-{int(time.time())}"
        st, tpl = call("POST", f"{REST}/templates", key, {"name": name, "imageName": a.image, "isServerless": True,
                                                           "containerDiskInGb": a.disk_gb, "env": env})
        if not ok(st): sys.exit(f"template {st}: {tpl}")
        st, ep = call("POST", f"{REST}/endpoints", key, {
            "templateId": tpl["id"], "name": name, "computeType": "GPU", "gpuTypeIds": GPU_4090, "gpuCount": 1,
            "workersMin": 0, "workersMax": 0, "idleTimeout": 5, "executionTimeoutMs": 900000, "flashboot": False,
            "networkVolumeId": a.volume_id})
        if not ok(st):
            call("DELETE", f"{REST}/templates/{tpl['id']}", key); sys.exit(f"endpoint {st}: {ep}")
        eid = ep["id"]
    out = Path(__file__).with_name("results") / f"{a.config}-{int(time.time())}.jsonl"
    # warm-up runs land in <out>.warmup.jsonl so summarize.py never counts them
    out.parent.mkdir(exist_ok=True)
    log(f"endpoint {eid} config {a.config} env {CONFIGS[a.config] or '{}'} -> {out}")
    try:
        for i in range(-a.warmup + 1, a.runs + 1):
            label = i if i >= 1 else f"warmup{i}"
            if not one_run(key, eid, label, out if i >= 1 else out.with_suffix(".warmup.jsonl")):
                break
            set_max_workers(key, eid, 0)
            log("scaling to 0, waiting for cold")
            if not wait_cold(key, eid):
                log("worker did not go away in 10 min; stopping"); break
    finally:
        call("PATCH", f"{REST}/endpoints/{eid}", key, {"workersMax": 0, "workersMin": 0})
        if not a.keep and tpl:
            call("DELETE", f"{REST}/endpoints/{eid}", key)
            call("DELETE", f"{REST}/templates/{tpl['id']}", key)
    log("done")


if __name__ == "__main__":
    main()
