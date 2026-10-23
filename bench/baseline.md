# Step 2 baseline runbook

Goal: a per-phase cold-start table for Qwen3-8B bf16 on a 4090 Serverless worker with
a network volume, three configurations, three cold runs each.

| Config | Loader | Env |
|--------|--------|-----|
| A | default (safetensors from volume) | `VLLM_CACHE_ROOT=/runpod-volume/vllm-cache` (compile cache warm, as in PR #351) |
| B | runai_streamer | A + `LOAD_FORMAT=runai_streamer`, extra config `{"concurrency": 32}` |
| C | default | A + `VLLM_ENABLE_STARTUP_PLAN=1` (second and third runs use the plan) |

Common env on the endpoint:

    MODEL_NAME=Qwen/Qwen3-8B
    HF_HOME=/runpod-volume/hf
    VLLM_CACHE_ROOT=/runpod-volume/vllm-cache
    VLLM_COLDSTART_PHASES_OUT=/runpod-volume/vllm-coldstart/phases.jsonl
    MAX_MODEL_LEN=8192

Image: worker-vllm `main` + `pip install /vllm-coldstart` (see worker/Dockerfile).
The plugin only observes; vLLM behaviour is unchanged.

Phases recorded (worker process):

1. `process_start` → `import_vllm`: Python import cost
2. `load_model`: weight load
3. `determine_available_memory`: profiling
4. `initialize_from_config`: KV allocation
5. `compile_or_warm_up_model`: compile (cache hit/miss) + kernel warmup + CUDA graph capture

Plus, from the handler (worker/handler.py): container start → handler import → engine
ready → first token on a 32-token prompt, written as `handler_*` lines.

Run 0 (warm-up, discarded) downloads the weights and fills the compile cache. Procedure per measured run: scale endpoint to 0 workers, wait 5 min, send one request, collect
`phases.jsonl`, append the run id. Delete the startup-plan dir between config A/B runs
so they stay comparable. `python bench/summarize.py phases.jsonl` prints the table.

## Running it

Build the image with the `image` GitHub Actions workflow (workflow_dispatch or a `v*` tag); it pushes to GHCR. Locally, from the repo root:

    docker build -f worker/Dockerfile -t ghcr.io/ashwinsreedhar28/vllm-coldstart:bench-v0.30.0 . && docker push ghcr.io/ashwinsreedhar28/vllm-coldstart:bench-v0.30.0

Then, with the API key in `~/.runpod.env` (`RUNPOD_API_KEY=...`):

    python3 bench/run_baseline.py --env-file ~/.runpod.env --list-volumes      # pick the PR #351 volume
    python3 bench/run_baseline.py --env-file ~/.runpod.env --image ghcr.io/ashwinsreedhar28/vllm-coldstart:bench-v0.30.0 --volume-id <id> --config A --runs 3
    python3 bench/run_baseline.py ... --config B --runs 3
    python3 bench/run_baseline.py ... --config C --runs 3      # run 1 writes the plan, runs 2-3 use it
    python3 bench/summarize.py bench/results/A-*.jsonl

Each run is one cold start on a 4090 (~2 min billed); nine runs is about $2-3 plus volume time.
The runner always leaves the endpoint at 0 workers and deletes it unless `--keep`.
