#!/bin/bash
set -e

MAX_LEN=4096
GPU_UTIL=0.9

if [ $# -lt 2 ]; then
  echo "Usage: $0 <MODEL_NAME> <PORT>"
  echo "Example: $0 Qwen/Qwen2.5-VL-3B-Instruct-AWQ 8001"
  exit 1
fi

MODEL_NAME=$1
PORT=$2

# nome leggibile per i log
RUN_NAME=$(echo "qwen-$(basename $MODEL_NAME)-$PORT" | tr '[:upper:]' '[:lower:]' | tr '/' '-')

LOG_FILE="/workspace/${RUN_NAME}.log"

echo "[INFO] Starting model ${MODEL_NAME} on port ${PORT}..."
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_NAME} \
  --trust-remote-code \
  --max-model-len ${MAX_LEN} \
  --gpu-memory-utilization ${GPU_UTIL} \
  --port ${PORT} > ${LOG_FILE} 2>&1 &

PID=$!
echo "[INFO] Server started with PID ${PID}, logs in ${LOG_FILE}"
echo "[INFO] Test: curl http://localhost:${PORT}/v1/models"