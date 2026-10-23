"""Step 5: KV checkpoint/resume KVConnector (stub; design in docs/design.md).

Base: vllm.distributed.kv_transfer.kv_connector.v1.example_connector.ExampleConnector
(451 lines, saves/loads KV per request to a directory). Changes needed:
- save on handle_preemptions() and request_finished() to the network volume,
  keyed by prompt-token hash so a different worker can find it
- get_num_new_matched_tokens() reports the resumable prefix so the scheduler
  skips prefill; start_load_kv() restores blocks layer by layer
- block-level manifest so partially written checkpoints are ignored
"""
