"""Step 3: capture-before-load (stub; design in docs/design.md).

Plan: patch Worker.load_model to call model_runner.load_model(load_dummy_weights=True),
then start a background thread that streams the real safetensors from the
network volume into the already-allocated parameter tensors via
GPUModelRunner.reload_weights(weights_iterator=...). The engine proceeds
through profile_run, KV allocation, compile (cache hit) and CUDA-graph capture
while weights arrive; the first execute_model blocks on the loader's done-event.

Open question to settle with a measurement before writing more: whether H2D
copies on a side stream from another thread are legal during capture_model
(torch.cuda.graph default capture mode is "global"). If not, the overlap
window is load_model..capture_model and capture waits for the join.
"""


def install() -> None:
    raise NotImplementedError("step 3 not started; see docs/design.md")
