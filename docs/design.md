# vllm-coldstart design

Status: step 1 done (2026-10-23). vLLM pinned at **v0.32.0**, the `ARG VLLM_VERSION`
in `runpod-workers/worker-vllm/Dockerfile` (`FROM vllm/vllm-openai:${VLLM_VERSION}`).

## Thesis

Every engine in the atlas assumes a warm, long-lived replica. Runpod Serverless is the
opposite: a worker is born cold, lives minutes, and can be preempted. vllm-coldstart makes
cold start and preemption the engine's problem through three vLLM plugins, each
measured against Qwen3-8B bf16 on a 4090 with a network volume (PR #351's setup,
baseline 68-71 s end to end).

## How vLLM v0.32.0 boots (verified in source)

`EngineCore._initialize_kv_caches` (vllm/v1/engine/core.py:254) drives the worker
(vllm/v1/worker/gpu_worker.py) in this order:

1. `init_device` (L360): CUDA init, distributed groups, memory snapshot.
2. `load_model(load_dummy_weights=False)` (L500): `GPUModelRunner.load_model`
   (gpu_model_runner.py:5312) creates the module on-device and calls the loader's
   `load_weights`. **The model must be complete before anything else runs.**
3. `determine_available_memory` (L528): `profile_run` = one dummy forward at
   `max_num_batched_tokens`; weight *values* are irrelevant, only shapes.
   Skippable with `VLLM_ENABLE_STARTUP_PLAN=1` (vllm/v1/worker/startup_plan.py,
   persists the KV-memory result under `$VLLM_CACHE_ROOT/startup_plan/` keyed by a
   fingerprint of config + device + vLLM version).
4. `initialize_from_config` (L735): allocate the KV cache.
5. `compile_or_warm_up_model` (L773): torch.compile per size (cache under
   `$VLLM_CACHE_ROOT`, the thing PR #351 persisted), `kernel_warmup`, then
   `capture_model` for CUDA graphs. **CUDA graphs are process-local and cannot be
   persisted**; this phase always runs on a cold worker.

Also present in 0.30.0 and relevant:

- `GPUModelRunner.reload_weights(weights_iterator=...)` (gpu_model_runner.py:5553):
  in-place weight reload with `initialize_layerwise_reload` / `finalize_layerwise_reload`
  (model_loader/reload/layerwise.py). Built for RL weight sync; it is the hook step 3 uses.
- `register_model_loader(load_format)` (model_loader/__init__.py:69) for a custom
  `BaseModelLoader` (`download_model`, `load_weights`).
- `RunaiModelStreamerLoader` (`--load-format runai_streamer`, extra config
  `concurrency`, `memory_limit`): concurrent chunked safetensors reads from local FS
  or object storage. The baseline should include it as a second loader.
- Weight-cache daemon (`model_loader/weight_cache/daemon.py`): per-GPU process that
  holds weights and serves CUDA IPC handles so a *restarting* engine on the *same host*
  maps them zero-copy. Warm restart only; a new Serverless container never has it.
- `KVConnectorBase_V1` (kv_transfer/kv_connector/v1/base.py): scheduler-side
  `get_num_new_matched_tokens` / `update_state_after_alloc` / `build_connector_meta` /
  `request_finished` / `handle_preemptions`; worker-side `start_load_kv` /
  `wait_for_layer_load` / `save_kv_layer` / `wait_for_save`. `ExampleConnector`
  (example_connector.py, 451 lines) is a per-request disk save/load reference.

## Answer to the step-1 risk

A partial-layer forward is not possible: `profile_run` needs the whole module and the
scheduler never runs before `compile_or_warm_up_model` returns. So "stream layers in
execution order so prefill starts early" is dead.

What replaces it is better: **capture-before-load**. Phases 3-5 need the model's
*shapes and buffers*, not its values. So:

1. `load_model(load_dummy_weights=True)`: parameters exist on-device with garbage.
2. Immediately start a background loader thread: iterate safetensors from the volume
   (Runai streamer iterator or our own mmap+pinned-staging reader) and feed
   `reload_weights(weights_iterator=...)`, which copies into the existing tensors.
3. Phases 3-5 run concurrently with the copies.
4. The first `execute_model` waits on the loader's done-event; a `/ready` probe
   reports not-ready until then.

The weight-load phase disappears from the critical path to the extent it overlaps.
With a compile-cache hit and a startup plan, the only serial work left on a cold worker
is container start, import, dummy init, KV alloc, kernel warmup and graph capture.

Open risk, to be measured first in step 3: `capture_model` uses `torch.cuda.graph`
with the default `capture_error_mode="global"` (vllm/compilation/cuda_graph.py:315).
Global capture mode makes "potentially unsafe" CUDA calls from *other threads* an
error, which may include the loader's H2D copies. If so, the loader joins before
capture and the overlap window is phases 3-4 plus compile; still a win, smaller.
`torch.cuda.graph` accepts `capture_error_mode="thread_local"`, but that is vLLM's call
site, not ours; an upstream flag is the fix if it matters.

Second risk: `process_weights_after_loading` runs once after the dummy load. For bf16
Qwen3-8B it is near a no-op; `finalize_layerwise_reload` re-runs the per-layer
processing after the real weights land. Any quantized model is out of scope until
this is verified.

## Steps (revised after step 1)

| # | Work | Gate | Spend |
|---|------|------|-------|
| 2 | Baseline: `vllm_coldstart.phase_timer` plugin + worker-level timers; three loaders (default, runai_streamer, default + `VLLM_ENABLE_STARTUP_PLAN=1`); 3 cold runs each on Serverless | Phase table with variance | ~$5 |
| 3 | `vllm_coldstart.capture_first` | Cold time-to-ready below baseline by at least the overlapped weight-load time; identical outputs to default load on 20 greedy prompts | ~$15 |
| 4 | Compile cache + startup plan as a volume asset keyed by (model, GPU SKU, vLLM version); measure the irreducible remainder (capture) | Fresh host: zero compile, zero profiling | ~$10 |
| 5 | `vllm_coldstart.kv_checkpoint` KVConnector | Kill worker mid-decode; resume elsewhere; greedy tokens identical | ~$20 |
| 6 | Speculative warm-up: design note only | Two pages with correctness argument | $0 |

Stop rule: if step 3 does not beat baseline, stop before 4 and 5.

## Out of scope

SGLang, TP/PP, quantized checkpoints, implementing speculative warm-up, anything from
Runpod internal systems.

## 2026-10-23 addenda

- Baseline (step 2) ran on v0.30.0; worker-vllm's base had moved to v0.32.0 by the time
  the branch was audited. Repinned to v0.32.0. `startup_plan.py` is byte-identical between
  the two; `Worker.load_model(load_dummy_weights=)`, `GPUModelRunner.reload_weights` and
  `register_model_loader` are unchanged; 0.32 adds `GPUModelRunner.load_model(model=...)`
  to hand in a pre-built module, worth a look for capture-before-load.
- Step 2 findings and numbers: bench/results/README.md.
- Startup plan: `VllmConfig.compute_hash()` ignores `gpu_memory_utilization`, so a cached
  plan is applied after the user changes the limit (vllm-project/vllm#60143, draft, fixes
  the fingerprint). The plan file is written as `<path>.tmp.<pid>` + `os.replace`, which
  collides across containers sharing a cache root. Default-on in worker-vllm waits for the
  fingerprint fix; the temp-file race is a separate upstream PR candidate.
