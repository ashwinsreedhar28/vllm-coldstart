# Baseline results (step 2), 2026-10-23

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
