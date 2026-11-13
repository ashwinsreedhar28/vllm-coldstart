"""Create one pod for the KV restore check and hand it to shwin-harness's pod_sweep.py --attach.

    python3 launch.py --mode full --volume-id <id> --out ~/development/shwin-harness/runs/kv-restore-p1
    python3 ~/development/shwin-harness/scripts/pod_sweep.py --attach <out> --wait 90

The pod has no start-up vLLM: job.sh starts, kills and restarts it. It removes itself after --max-hours.
"""
import argparse, io, json, os, sys, tarfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ.get("SHWIN_HARNESS", HERE.parents[2] / "shwin-harness")) / "scripts"))
import pod_sweep as ps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["full", "net", "restore", "blocks"], required=True)
ap.add_argument("--volume-id", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--gpu", action="append", default=[])
ap.add_argument("--max-hourly", type=float, default=0.80)
ap.add_argument("--max-hours", type=float, default=1.2)
ap.add_argument("--disk-gb", type=int, default=80)
ap.add_argument("--key-file", type=Path, default=Path.home() / "runpod-key" / "key.txt")
a = ap.parse_args()

ns = argparse.Namespace(model="x", max_model_len=1, gpu_memory_utilization=0.9, max_num_seqs=1, vllm_arg=[],
                        max_hours=a.max_hours, grace_minutes=8)
cmd = "\n".join(line for line in ps.start_command(ns).splitlines() if "vllm serve" not in line)
pod = ps.Pod(ps.read_key(a))
body = {"name": f"kv-restore-{a.mode}-{time.strftime('%m%d-%H%M')}", "imageName": "vllm/vllm-openai:latest",
        "gpuTypeIds": a.gpu or ["NVIDIA GeForce RTX 4090"], "gpuCount": 1, "containerDiskInGb": a.disk_gb,
        "networkVolumeId": a.volume_id, "volumeMountPath": "/workspace", "ports": ["8000/http"],
        "env": {"VLLM_API_KEY": ps.secrets.token_hex(16), "JOB_TOKEN": pod.token},
        "dockerEntrypoint": ["bash", "-c"], "dockerStartCmd": [cmd]}
r = pod.create(body)
pod.id = r["id"]
price = float(r.get("costPerHr") or 0)
print(f"pod {pod.id} at ${price:.2f}/h on {(r.get('machine') or {}).get('gpuTypeId')}", flush=True)
if price > a.max_hourly:
    print(f"REFUSED: ${price:.2f}/h is above --max-hourly {a.max_hourly}")
    pod.delete()
    sys.exit(2)
up = a.out / "upload"
up.mkdir(parents=True, exist_ok=True)
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    tar.add(HERE / "bench.py", arcname="bench.py")
(up / "bundle.tgz").write_bytes(buf.getvalue())
(up / "setup.sh").write_text("true\n")
(up / "job_sweep.sh").write_text((HERE / "job.sh").read_text().replace("__MODE__", a.mode))
sf = a.out / "pod.json"
sf.write_text(json.dumps({"id": pod.id, "token": pod.token, "price": price, "t0": time.time(), "seen": 0, "uploaded": False}))
sf.chmod(0o600)
print(f"pod created; continue with: pod_sweep.py --attach {a.out}")
