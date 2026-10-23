"""Per-phase startup timing for the vLLM v0.30.0 GPU worker.

Wraps the four worker methods EngineCore calls in order
(vllm/v1/engine/core.py, _initialize_kv_caches):

    load_model -> determine_available_memory -> initialize_from_config
               -> compile_or_warm_up_model

and records wall time for each, plus process-start and vllm-import offsets,
as JSON lines. Nothing here changes behaviour; it only observes. vLLM's own
"Loading weights took" log line is not trusted (vllm-project/vllm#63019).
"""
from __future__ import annotations

import functools
import json
import os
import time

_T0 = time.time()


def _proc_start_time() -> float:
    """Process start wall time from /proc (Linux only); falls back to import time."""
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().split(")")[-1].split()[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        clk = os.sysconf("SC_CLK_TCK")
        boot = time.time() - uptime
        return boot + start_ticks / clk
    except Exception:  # noqa: BLE001
        return _T0


def _write(path: str, rec: dict) -> None:
    rec = {"pid": os.getpid(), "rank": os.environ.get("RANK", "0"), **rec}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _wrap(cls, name: str, out: str) -> None:
    orig = getattr(cls, name)
    if getattr(orig, "_vllm_coldstart_wrapped", False):
        return

    @functools.wraps(orig)
    def timed(self, *a, **kw):
        t = time.time()
        try:
            return orig(self, *a, **kw)
        finally:
            _write(out, {"phase": name, "start": t, "end": time.time(), "secs": round(time.time() - t, 3)})

    timed._vllm_coldstart_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, name, timed)


def install(out: str) -> None:
    from vllm.v1.worker.gpu_worker import Worker  # import is itself a phase; record it

    _write(out, {"phase": "process_start", "start": _proc_start_time(), "end": _proc_start_time(), "secs": 0})
    _write(out, {"phase": "import_vllm", "start": _proc_start_time(), "end": time.time(),
                 "secs": round(time.time() - _proc_start_time(), 3)})
    for name in ("load_model", "determine_available_memory", "initialize_from_config",
                 "compile_or_warm_up_model"):
        _wrap(Worker, name, out)
