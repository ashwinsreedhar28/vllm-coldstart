# vLLM v0.30.0 interface notes (file:line as of tag v0.30.0)

| What | Where |
|------|-------|
| Plugin entry point group | `vllm.general_plugins` (vllm/plugins/__init__.py:18); loaded in every process, must be idempotent |
| Worker boot sequence | vllm/v1/engine/core.py:254 `_initialize_kv_caches`; vllm/v1/worker/gpu_worker.py `load_model`:500, `determine_available_memory`:528, `initialize_from_config`:735, `compile_or_warm_up_model`:773 |
| Dummy load | `Worker.load_model(load_dummy_weights=True)` → `GPUModelRunner.load_model(load_dummy_weights=True)` gpu_model_runner.py:5312 |
| In-place reload | `GPUModelRunner.reload_weights(weights_iterator, weights_path, is_checkpoint_format)` gpu_model_runner.py:5553 |
| Custom loader | `@register_model_loader("name")` on a `BaseModelLoader` subclass, model_loader/__init__.py:69; `--load-format name` |
| Streaming loader | `RunaiModelStreamerLoader`, model_loader/runai_streamer_loader.py; extra config keys `distributed`, `concurrency`, `memory_limit` |
| Compile cache | `VLLM_CACHE_ROOT` (default `~/.cache/vllm`), vllm/envs.py:34; `VLLM_DISABLE_COMPILE_CACHE` |
| Startup plan | `VLLM_ENABLE_STARTUP_PLAN=1`, vllm/v1/worker/startup_plan.py; skips memory profiling and cudagraph memory estimation when fingerprint and free memory match |
| CUDA graph capture | vllm/compilation/cuda_graph.py:315 `torch.cuda.graph(cudagraph, pool=..., stream=current_stream())`, default global capture mode |
| Weight cache daemon | model_loader/weight_cache/daemon.py; `--load-format ipc_cache`; same-host restarts only |
| KV connector base | vllm/distributed/kv_transfer/kv_connector/v1/base.py: scheduler side `get_num_new_matched_tokens`:489, `update_state_after_alloc`:524, `build_connector_meta`:550, `request_finished`:582, `handle_preemptions`:305; worker side `start_load_kv`:313, `wait_for_layer_load`:331, `save_kv_layer`:345, `wait_for_save`:367 |
| Disk connector reference | kv_connector/v1/example_connector.py (451 lines) |
