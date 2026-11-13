"""Serverless handler for the session-gap run. Delivered to the worker through the template's start command
(see drive.py), so no image rebuild is needed. Reports what a client cannot see: which process answered,
how old it is, and how many prompt tokens were served from cache.
"""
import json, os, shutil, time, uuid

T_IMPORT = time.time()
BOOT_ID = uuid.uuid4().hex[:8]  # changes whenever this process is a new one; survives a paused-container revival
LOG = "/tmp/worker.log"
STATE = {"engine": None, "engine_ready_s": None, "requests": 0, "engine_error": None}


def _engine():
    if STATE["engine"] is None and STATE["engine_error"] is None:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        t = time.time()
        kw = {}
        if os.environ.get("KV_TRANSFER_CONFIG"):
            from vllm.config import KVTransferConfig
            kw["kv_transfer_config"] = KVTransferConfig(**json.loads(os.environ["KV_TRANSFER_CONFIG"]))
        try:
            args = AsyncEngineArgs(model=os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B"),
                                   max_model_len=int(os.environ.get("MAX_MODEL_LEN", "32000")),
                                   gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.9")),
                                   max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "8")),  # hybrid models need one Mamba block per sequence
                                   enable_prefix_caching=True, **kw)
            STATE["engine"] = AsyncLLM.from_engine_args(args)
            STATE["engine_ready_s"] = round(time.time() - t, 2)
        except Exception as e:  # report it to the client instead of crash-looping a billed worker
            STATE["engine_error"] = repr(e)[:300] + " | " + _log_tail()
    return STATE["engine"]


def _log_tail(n=2500):
    """The EngineCore child writes the root cause to stderr, which __main__ redirected to LOG."""
    try:
        lines = [l for l in open(LOG, errors="replace").read().splitlines() if "ERROR" in l or "Error" in l]
        return " // ".join(l[-300:] for l in lines if "File " not in l and "^^^" not in l)[-n:]
    except Exception as e:  # noqa: BLE001
        return repr(e)[:200]


def _env():
    out = {"worker_id": os.environ.get("RUNPOD_POD_ID"), "boot_id": BOOT_ID, "boot_age_s": round(time.time() - T_IMPORT, 1),
           "engine_ready_s": STATE["engine_ready_s"], "requests_this_boot": STATE["requests"],
           "volume": os.path.isdir("/runpod-volume"), "kv_config": bool(os.environ.get("KV_TRANSFER_CONFIG"))}
    try:
        out["shm_free_mb"] = shutil.disk_usage("/dev/shm").free // 2**20
        import torch
        out["gpu"] = torch.cuda.get_device_name(0)
        import vllm
        out["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        out["env_error"] = repr(e)[:200]
    return out


def _volume(clean):
    """Bytes per top-level directory of the network volume; optionally remove this benchmark's KV block files."""
    root = "/runpod-volume"; out = {}
    if clean and os.path.isdir(f"{root}/kv_gap"):
        shutil.rmtree(f"{root}/kv_gap", ignore_errors=True)
    try:
        for d in sorted(os.listdir(root)):
            n = b = 0
            for dp, _, fs in os.walk(os.path.join(root, d)):
                for f in fs:
                    try:
                        b += os.lstat(os.path.join(dp, f)).st_blocks * 512; n += 1
                    except OSError:
                        pass
            out[d] = {"files": n, "gb": round(b / 2**30, 2)}
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:200]
    return {"volume_dirs": out}


async def handler(job):
    from vllm import SamplingParams
    t_in = time.time()
    inp = job.get("input", {})
    if inp.get("probe"):  # no engine: what does a worker look like before any model work
        return {"probe": True, **_volume(inp.get("clean_kv")), **_env()}
    eng = _engine()
    if eng is None:
        return {"error": STATE["engine_error"], **_env()}
    STATE["requests"] += 1
    sp = SamplingParams(max_tokens=int(inp.get("max_tokens", 8)), temperature=0.0)
    t = time.time(); first = None; text = ""; cached = None; n_prompt = None
    async for out in eng.generate(inp["prompt"], sp, request_id=job["id"]):
        if first is None:
            first = time.time()
        text = out.outputs[0].text
        cached = getattr(out, "num_cached_tokens", cached)
        if getattr(out, "prompt_token_ids", None):
            n_prompt = len(out.prompt_token_ids)
    return {"text": text, "ttft_s": round((first or time.time()) - t, 3), "gen_s": round(time.time() - t, 3),
            "engine_wait_s": round(t - t_in, 2), "prompt_tokens": n_prompt, "cached_tokens": cached,
            "t_handler_in": t_in, "t_handler_out": time.time(), **_env()}


if __name__ == "__main__":  # the EngineCore child re-imports this module; only the parent may start the worker
    import runpod
    fd = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 1); os.dup2(fd, 2)  # children inherit it; a failed engine start can then be reported in the job output
    runpod.serverless.start({"handler": handler})
