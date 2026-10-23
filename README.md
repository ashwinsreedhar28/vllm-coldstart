# vllm-coldstart

Serverless-native cold start and preemption plugins for [vLLM](https://github.com/vllm-project/vllm),
built for Runpod Serverless. Pinned to vLLM v0.30.0, the base of
[worker-vllm](https://github.com/runpod-workers/worker-vllm).

Every production engine assumes a warm, long-lived replica. A Serverless worker is born
cold and can be preempted. vllm-coldstart makes both the engine's problem:

| Module | Mechanism | Status |
|--------|-----------|--------|
| `vllm_coldstart.phase_timer` | Per-phase cold-start timing of the vLLM worker | baseline tooling, done |
| `vllm_coldstart.capture_first` | Dummy-load the model, stream real weights in the background while profiling, compile and CUDA-graph capture run | design, see docs/design.md |
| `vllm_coldstart.kv_checkpoint` | KVConnector that checkpoints a request's KV to the network volume on preemption and resumes on another worker | design |

Baseline target: Qwen3-8B bf16 on an RTX 4090 with a network volume, where a stock
worker-vllm cold start is 68-71 s with a warm compile cache
([worker-vllm #351](https://github.com/runpod-workers/worker-vllm/pull/351)).

## Layout

    vllm-coldstart/   vLLM general plugin (entry point `vllm.general_plugins`)
    worker/      minimal Runpod handler + Dockerfile for experiments
    bench/       runbook and summarizer for the baseline
    docs/        design.md (thesis, boot sequence, risks), interfaces.md (file:line map)
    tests/       unit tests that run without a GPU

## Run the unit tests

    uv venv && uv pip install -e ".[dev]" && uv run pytest -q

## License

MIT
