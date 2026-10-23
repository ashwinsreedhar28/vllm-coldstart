"""vllm-coldstart: serverless-native cold start and preemption plugins for vLLM.

Three mechanisms, each a separate module, each measurable on its own:

- phase_timer   per-phase cold-start timing of the vLLM worker (baseline, step 2)
- capture_first dummy-load the model, stream real weights in the background
                while profiling, compile and CUDA-graph capture run (step 3)
- kv_checkpoint KVConnector that checkpoints a request's KV blocks to the
                network volume on preemption and resumes elsewhere (step 5)

Pinned against vLLM v0.30.0 (the worker-vllm base image).
"""

__version__ = "0.0.1"
