# Baseline results (step 2), 2026-10-23

## All configs, median of 3 measured cold starts (seconds)

| config | engine ready | load_model | profile | warm-up + capture | handler import |
|---|---|---|---|---|---|
| A default | 63.3 | 31.5 | 6.9 | 7.8 | 19.1 |
| B runai_streamer | 62.3 | 27.4 (15.6-293) | 7.3 | 7.8 | 18.9 |
| C `VLLM_ENABLE_STARTUP_PLAN=1` | 54.0 | 25.9 | 2.3 | 7.4 | 18.3 |
| D `RUNPOD_SKIP_AUTO_SYSTEM_CHECKS=true` | 52.3 | 31.9 | 6.9 | 7.5 | 18.8 |
| E `safetensors_load_strategy=prefetch` | 51.8 (35.4-97.6) | 20.0 (5.8-22.3) | 6.4 | 6.7 | 9.0 |
| F `safetensors_load_strategy=eager` | 53.6 (53.3-55.6) | 24.9 (24.3-27.3) | 6.4 | 5.4 | 8.4 |

A-D ran image `bench-d77bef1`; E-F ran `bench-5d3123e` (adds the load-strategy passthrough to the handler).
D: one `import_vllm` record per run (EngineCore forked, no re-import); A-C, E-F: two (spawned).
E run 1 recompiled (profile 42.0 s) for an unidentified reason; `LoadConfig.compute_hash()` has no factors, so not the strategy.
Open: handler import is 8.4-9.5 s on every E/F run and 17-20 s on every A-D run. The images differ by one handler
line and the pip install time. Needs a controlled rerun of config A on `bench-5d3123e`.


Qwen3-8B bf16, vLLM v0.30.0, image `ghcr.io/ashwinsreedhar28/vllm-coldstart:bench-d77bef1`,
Runpod Serverless, network volume `qin62zym4t` (US-TX-3), GPU filter RTX 4090 class.
Three measured cold starts per config after one discarded warm-up. Seconds, median of 3.

| phase | A default | B runai_streamer | C startup plan |
|---|---|---|---|
| platform delay (submit → handler up) | 67.2 | 29.5 | 26.4 |
| handler Python import | 19.1 | 18.9 | 17.9 |
| vLLM import in EngineCore process | 11.3 | 11.6 | 11.1 |
| load_model | 31.5 | 27.4 | 23.2 |
| determine_available_memory | 6.9 | 7.3 | 2.3 |
| compile_or_warm_up_model | 7.8 | 7.8 | 7.2 |
| engine ready (handler clock) | 63.3 | 62.3 | 54.0 |
| TTFT after ready | 0.09 | 0.08 | 0.08 |

load_model across all nine runs: 7.0, 15.6, 23.2, 27.3, 27.4, 29.9, 31.5, 32.5, 292.9.
The spread is the volume, not the loader.

Startup-plan verification (new endpoint `vllm-faxpggoqrljcb9`, same volume, 2 cold starts,
no warm-up): determine_available_memory 2.26 s and 2.91 s. The plan written by the first
endpoint's workers was applied by the second endpoint's workers, so it survives a host change.

Findings:
1. Weight load is volume-bound and highly variable; loader choice is noise against it.
2. `VLLM_ENABLE_STARTUP_PLAN=1` saves 4.6 s per cold start (profile 6.9 → 2.3 s), every run.
3. 30 s of the 63 is Python import; 11 s of that is a re-import in the spawned EngineCore,
   forced because Runpod's SDK fitness checks initialize CUDA in the handler process.
4. With the compile cache warm, compile is not a phase: warm-up + CUDA graph capture is ~7.8 s.

Raw: `A-*.jsonl`, `B-*.jsonl`, `C-*.jsonl` (one line per run); `*.warmup.jsonl` discarded boots;
`A-warmup-boot.json` the first boot on an empty volume (engine ready 164 s).
Reproduce: `python3 bench/run_baseline.py --endpoint-id <id> --config A --runs 3`.
