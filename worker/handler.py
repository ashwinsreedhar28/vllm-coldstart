"""Minimal Runpod Serverless handler around vLLM's AsyncLLM with handler-level timers.

Used only for the baseline and for vllm-coldstart experiments; the production worker is
runpod-workers/worker-vllm. Writes handler_* phases to VLLM_COLDSTART_PHASES_OUT.
"""
import asyncio, json, os, time

T_START = time.time()
OUT = os.environ.get("VLLM_COLDSTART_PHASES_OUT", "/runpod-volume/vllm-coldstart/phases.jsonl")


def _log(phase, t0, t1):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps({"phase": phase, "start": t0, "end": t1, "secs": round(t1 - t0, 3), "pid": os.getpid()}) + "\n")


import runpod  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402

if __name__ == "__main__":
    _log("handler_import", T_START, time.time())

engine = None


async def _get_engine():
    global engine
    if engine is None:
        t = time.time()
        args = AsyncEngineArgs(
            model=os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B"),
            max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
            load_format=os.environ.get("LOAD_FORMAT", "auto"),
            model_loader_extra_config=json.loads(os.environ.get("LOADER_EXTRA", "{}") or "{}"),
            gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.9")),
            safetensors_load_strategy=os.environ.get("SAFETENSORS_LOAD_STRATEGY") or None,
        )
        engine = AsyncLLM.from_engine_args(args)
        _log("handler_engine_ready", t, time.time())
    return engine


async def handler(job):
    eng = await _get_engine()
    inp = job.get("input", {})
    prompt = inp.get("prompt", "Explain paged attention in one paragraph.")
    sp = SamplingParams(max_tokens=int(inp.get("max_tokens", 64)), temperature=0.0)
    t = time.time()
    first = None
    text = ""
    async for out in eng.generate(prompt, sp, request_id=job["id"]):
        if first is None:
            first = time.time()
            _log("handler_first_token", t, first)
        text = out.outputs[0].text
    return {"text": text, "ttft_s": round(first - t, 3) if first else None, "phases": _phases_this_boot(),
            "gpu": _gpu_name(), "worker_id": os.environ.get("RUNPOD_POD_ID")}


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None


def _phases_this_boot():
    """Phase records written since this handler process started (all pids: worker
    processes are children spawned after T_START)."""
    try:
        recs = [json.loads(l) for l in open(OUT) if l.strip()]
    except FileNotFoundError:
        return []
    return [r for r in recs if r.get("start", 0) >= T_START - 1 and r["phase"] != "process_start"]


if __name__ == "__main__":
    # vLLM forces the `spawn` start method here (Runpod's fitness checks initialize
    # CUDA in this process), and spawn re-imports this module in the EngineCore
    # child. Without this guard the child starts a second Serverless worker and
    # the container dies ~25 s after "Started." (seen on the first baseline run).
    runpod.serverless.start({"handler": handler})
