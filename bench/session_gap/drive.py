"""Session-gap run on Runpod Serverless: one growing session, a fixed gap before each turn.

    python3 drive.py create --arm fb1-idle5 --flashboot 1 --idle 5 --volume-id <id> --out runs/fb1-idle5
    python3 drive.py step --out runs/fb1-idle5 --budget 150      # repeat until it prints DONE
    python3 drive.py teardown --out runs/fb1-idle5

`step` is resumable: it does as much of the plan as fits in --budget seconds and saves state, so it can be
driven from a shell that cannot keep a process alive. Stdlib only.
"""
import argparse, base64, json, os, random, sys, time, urllib.error, urllib.request
from pathlib import Path

REST, JOBS = "https://rest.runpod.io/v1", "https://api.runpod.ai/v2"
HERE = Path(__file__).resolve().parent
WORDS = ("alpha bravo cedar delta ember fjord grove harbor island jasper kettle lumen meadow nectar onyx prairie quartz "
         "river summit timber umbra valley willow xenon yarrow zephyr anchor beacon copper dune echo falcon garnet").split()
IMAGE = "vllm/vllm-openai:v0.32.0"  # public base of worker-vllm; the runpod SDK is installed at start


def key(a):
    return Path(a.key_file).expanduser().read_text().strip()


def call(method, url, k, body=None, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Authorization": f"Bearer {k}", "Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:600]
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, str(e)


def prompt(session: str, turn: int, base_items: int, turn_items: int) -> str:
    """About 13 tokens per item with the Qwen tokenizer. Turn k's prompt extends turn k-1's, minus the last line."""
    def chunk(tag, n):
        rng = random.Random(f"gap-{session}-{tag}")
        return " ".join(f"{i}:{rng.choice(WORDS)}-{rng.randint(0, 9999)}" for i in range(n))
    text = f"Session {session}. Log follows. " + chunk("base", base_items)
    for j in range(1, turn + 1):
        text += f"\nTool output {j}: " + chunk(f"t{j}", turn_items)
    return text + "\nReply with the single word OK."


def load(out): return json.loads((Path(out) / "state.json").read_text())
def save(out, s): (Path(out) / "state.json").write_text(json.dumps(s, indent=1))


def create(a):
    k = key(a); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = {"MODEL_NAME": a.model, "HF_HOME": "/runpod-volume/hf", "VLLM_CACHE_ROOT": "/runpod-volume/vllm-cache",
           "MAX_MODEL_LEN": str(a.max_model_len)}
    if a.fs:
        env["KV_TRANSFER_CONFIG"] = json.dumps({"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {
            "spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": a.cpu_gb * 2**30, "block_size": a.block_size,
            "secondary_tiers": [{"type": "fs", "root_dir": f"/runpod-volume/kv_gap/{a.arm}"}]}})
    b64 = base64.b64encode((HERE / "handler.py").read_bytes()).decode()
    name = f"session-gap-{a.arm}-{int(time.time())}"
    st, tpl = call("POST", f"{REST}/templates", k, {"name": name, "imageName": a.image, "isServerless": True, "containerDiskInGb": 30, "env": env,
                                                   "dockerEntrypoint": ["bash", "-c"],
                                                   "dockerStartCmd": [f"pip install -q runpod; echo {b64} | base64 -d > /h.py && exec python3 /h.py"]})
    if not 200 <= st < 300:
        sys.exit(f"template {st}: {tpl}")
    st, ep = call("POST", f"{REST}/endpoints", k, {"templateId": tpl["id"], "name": name, "computeType": "GPU", "gpuTypeIds": a.gpu,
                                                   "gpuCount": 1, "workersMin": 0, "workersMax": 0, "idleTimeout": a.idle,
                                                   "executionTimeoutMs": 900000, "flashboot": bool(a.flashboot), "networkVolumeId": a.volume_id})
    if not 200 <= st < 300:
        call("DELETE", f"{REST}/templates/{tpl['id']}", k); sys.exit(f"endpoint {st}: {ep}")
    plan = [{"turn": i, "gap": g} for i, g in enumerate([None] + [float(x) for x in a.gaps.split(",")])]
    if a.probe:
        plan = [{"turn": -1, "gap": None, "probe": True}] + plan
    save(out, {"arm": a.arm, "eid": ep["id"], "tpl": tpl["id"], "session": f"{a.arm}-{a.session}", "plan": plan, "i": 0, "job": None,
               "last_done": None, "base_items": a.base_items, "turn_items": a.turn_items, "max_workers": a.max_workers,
               "endpoint": {x: ep.get(x) for x in ("idleTimeout", "flashboot", "gpuTypeIds", "networkVolumeId", "workersMax")},
               "template_echo": {x: tpl.get(x) for x in ("dockerEntrypoint", "dockerStartCmd", "imageName") if x != "dockerStartCmd"} | {"start_cmd_len": len(str(tpl.get("dockerStartCmd")))}})
    print(f"created endpoint {ep['id']} template {tpl['id']} arm {a.arm}: {json.dumps(load(out)['endpoint'])}")


def health(k, eid):
    st, h = call("GET", f"{JOBS}/{eid}/health", k, timeout=20)
    return h.get("workers") if isinstance(h, dict) else {"error": str(h)[:100]}


def step(a):
    k = key(a); s = load(a.out); eid = s["eid"]; t_end = time.time() + a.budget
    rows = Path(a.out) / "rows.jsonl"
    while time.time() < t_end:
        if s["i"] >= len(s["plan"]):
            call("PATCH", f"{REST}/endpoints/{eid}", k, {"workersMax": 0, "workersMin": 0})
            print("DONE"); save(a.out, s); return
        p = s["plan"][s["i"]]
        if s["job"] is None:
            due = (s["last_done"] or 0) + (p["gap"] or 0)
            if due - time.time() > t_end - time.time():
                save(a.out, s); time.sleep(max(0.0, t_end - time.time()))  # use the whole budget, then report
                print(f"WAIT {due - time.time():.0f} s until turn {p['turn']}"); return
            time.sleep(max(0.0, due - time.time()))
            if s["i"] == 0:
                call("PATCH", f"{REST}/endpoints/{eid}", k, {"workersMax": s["max_workers"]}); time.sleep(4)
            body = {"input": {"probe": True} if p.get("probe") else {"prompt": prompt(s["session"], p["turn"], s["base_items"], s["turn_items"])}}
            h0 = health(k, eid)
            for _ in range(20):
                t0 = time.time()
                st, job = call("POST", f"{JOBS}/{eid}/run", k, body)
                if 200 <= st < 300 or "ENDPOINT_PAUSED" not in str(job):
                    break
                time.sleep(3)
            if not 200 <= st < 300:
                print(f"submit failed {st} {str(job)[:300]}"); save(a.out, s); return
            s["job"] = {"id": job["id"], "t_submit": t0, "health_at_submit": h0}
            save(a.out, s)
        st, r = call("GET", f"{JOBS}/{eid}/status/{s['job']['id']}", k, timeout=20)
        if 200 <= st < 300 and r.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            now = time.time()
            row = {"arm": s["arm"], "turn": p["turn"], "gap_s": p["gap"], "status": r["status"], "wall_s": round(now - s["job"]["t_submit"], 2),
                   "delay_s": round((r.get("delayTime") or 0) / 1000, 2), "exec_s": round((r.get("executionTime") or 0) / 1000, 2),
                   "job_worker": r.get("workerId"), "t_submit": s["job"]["t_submit"], "t_done": now,
                   "health_at_submit": s["job"]["health_at_submit"], "health_at_done": health(k, eid), "out": r.get("output") if r["status"] == "COMPLETED" else str(r)[:800]}
            with open(rows, "a") as f:
                f.write(json.dumps(row) + "\n")
            o = row["out"] if isinstance(row["out"], dict) else {}
            print(f"turn {p['turn']} gap {p['gap']} {r['status']} wall {row['wall_s']} delay {row['delay_s']} exec {row['exec_s']} ttft {o.get('ttft_s')} "
                  f"cached {o.get('cached_tokens')}/{o.get('prompt_tokens')} worker {o.get('worker_id')} boot {o.get('boot_id')} age {o.get('boot_age_s')} "
                  f"req# {o.get('requests_this_boot')} ready {o.get('engine_ready_s')} gpu {o.get('gpu')} err {str(o.get('error'))[:300] if o.get('error') else ''}", flush=True)
            s.update(i=s["i"] + 1, job=None, last_done=now); save(a.out, s)
            if r["status"] != "COMPLETED" or o.get("error"):
                call("PATCH", f"{REST}/endpoints/{eid}", k, {"workersMax": 0, "workersMin": 0})
                print("STOPPED: scaled to 0 after a failed turn"); return
        else:
            time.sleep(1.0)
    save(a.out, s); print(f"[budget used; job {'running' if s['job'] else 'none'}; next turn index {s['i']}]")


def teardown(a):
    k = key(a); s = load(a.out)
    call("PATCH", f"{REST}/endpoints/{s['eid']}", k, {"workersMax": 0, "workersMin": 0})
    print("endpoint delete", call("DELETE", f"{REST}/endpoints/{s['eid']}", k)[0], "template delete", call("DELETE", f"{REST}/templates/{s['tpl']}", k)[0])
    st, eps = call("GET", f"{REST}/endpoints", k)
    print("endpoints left:", [e["id"] for e in eps] if isinstance(eps, list) else eps)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
for n in ("create", "step", "teardown"):
    q = sub.add_parser(n); q.add_argument("--out", required=True); q.add_argument("--key-file", default="~/runpod-key/key.txt")
c = sub.choices["create"]
c.add_argument("--arm", required=True); c.add_argument("--flashboot", type=int, default=1); c.add_argument("--idle", type=int, default=5)
c.add_argument("--fs", type=int, default=0); c.add_argument("--block-size", type=int, default=512); c.add_argument("--cpu-gb", type=int, default=6)
c.add_argument("--volume-id", required=True); c.add_argument("--image", default=IMAGE); c.add_argument("--model", default="Qwen/Qwen3-8B")
c.add_argument("--max-model-len", type=int, default=28000); c.add_argument("--gaps", default="0,10,60,300"); c.add_argument("--session", default="s1")
c.add_argument("--base-items", type=int, default=1500); c.add_argument("--turn-items", type=int, default=40); c.add_argument("--max-workers", type=int, default=1)
c.add_argument("--probe", type=int, default=0); c.add_argument("--gpu", action="append")
sub.choices["step"].add_argument("--budget", type=float, default=150)
a = ap.parse_args()
if a.cmd == "create" and not a.gpu:
    a.gpu = ["NVIDIA GeForce RTX 4090"]
{"create": create, "step": step, "teardown": teardown}[a.cmd](a)
