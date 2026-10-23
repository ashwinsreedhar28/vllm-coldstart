"""vLLM general-plugin entry point.

vLLM calls this in every process it spawns (API server, EngineCore, workers)
before the model loads. It must be idempotent and cheap. Which mechanisms it
turns on is controlled by env vars so the same wheel works in every step:

    VLLM_COLDSTART_PHASES_OUT=/runpod-volume/vllm-coldstart/phases.jsonl
        write one JSON line per startup phase (step 2 baseline)
    VLLM_COLDSTART_CAPTURE_FIRST=1
        dummy-load, then stream real weights in the background (step 3)
"""
import os


def register() -> None:
    if os.environ.get("VLLM_COLDSTART_PHASES_OUT"):
        from vllm_coldstart.phase_timer import install
        install(os.environ["VLLM_COLDSTART_PHASES_OUT"])
    if os.environ.get("VLLM_COLDSTART_CAPTURE_FIRST") == "1":
        from vllm_coldstart.capture_first import install
        install()
