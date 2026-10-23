# Baseline/experiment image: upstream worker-vllm base + vllm-coldstart plugin.
ARG VLLM_VERSION=v0.32.0
FROM vllm/vllm-openai:${VLLM_VERSION}
RUN pip install --no-cache-dir runpod
COPY . /vllm-coldstart
RUN pip install --no-cache-dir -e /vllm-coldstart
ENV VLLM_COLDSTART_PHASES_OUT=/runpod-volume/vllm-coldstart/phases.jsonl
ENTRYPOINT []
CMD ["python3", "/vllm-coldstart/worker/handler.py"]
