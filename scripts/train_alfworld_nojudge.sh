#!/bin/bash
# =============================================================================
# train_alfworld.sh — Spin up vLLM servers + run ALFWorld MemRL training
#
# Usage:
#   bash scripts/train_alfworld.sh
# =============================================================================
set -e

# ── paths & binaries ─────────────────────────────────────────────────────────
PYTHON=/home/qinbowen/miniconda3/envs/memrl/bin/python
VLLM=/home/qinbowen/miniconda3/envs/dschat/bin/vllm
PROJECT=/home/qinbowen/MemRL
HF_HOME=/huggingface

# ── LLM server (tensor parallel across 2 GPUs) ──────────────────────────────
LLM_MODEL=/home/qinbowen/models/Qwen3-4B-Instruct-2507
LLM_PORT=8000
LLM_NAME=Qwen3-4B-Instruct-2507
LLM_GPU=0,1
LLM_TP=2
LLM_LOG=/tmp/vllm_llm.log

# ── Embedding server ─────────────────────────────────────────────────────────
EMBED_MODEL=/home/qinbowen/models/Qwen3-Embedding-4B
EMBED_PORT=8001
EMBED_NAME=Qwen3-Embedding-4B
EMBED_GPU=0,1
EMBED_TP=2
EMBED_LOG=/tmp/vllm_embed.log

# ── Training ─────────────────────────────────────────────────────────────────
TRAIN_LOG=$PROJECT/alfworld_nojudge.log
CONFIG=$PROJECT/configs/rl_alf_config.nojudge.yaml

# ── helpers ──────────────────────────────────────────────────────────────────
wait_for_server() {
    local name=$1 port=$2
    echo "[$(date +%T)] Waiting for $name on port $port..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:$port/health > /dev/null 2>&1; then
            echo "[$(date +%T)] $name is ready."
            return 0
        fi
        sleep 5
    done
    echo "[$(date +%T)] ERROR: $name did not start within 5 minutes."
    exit 1
}

cleanup() {
    echo "[$(date +%T)] Shutting down servers..."
    kill $LLM_PID $EMBED_PID 2>/dev/null || true
    wait $LLM_PID $EMBED_PID 2>/dev/null || true
    echo "[$(date +%T)] Servers stopped."
}
trap cleanup EXIT

# ── 1. start LLM server ─────────────────────────────────────────────────────
echo "[$(date +%T)] Starting LLM server ($LLM_NAME) on GPU $LLM_GPU (tp=$LLM_TP), port $LLM_PORT..."
CUDA_VISIBLE_DEVICES=$LLM_GPU HF_HOME=$HF_HOME $VLLM serve $LLM_MODEL \
    --host 0.0.0.0 \
    --port $LLM_PORT \
    --served-model-name $LLM_NAME \
    --tensor-parallel-size $LLM_TP \
    --gpu-memory-utilization 0.85 \
    > $LLM_LOG 2>&1 &
LLM_PID=$!
echo "[$(date +%T)] LLM server PID: $LLM_PID"

# ── 2. start embedding server ───────────────────────────────────────────────
echo "[$(date +%T)] Starting embedding server ($EMBED_NAME) on GPU $EMBED_GPU, port $EMBED_PORT..."
CUDA_VISIBLE_DEVICES=$EMBED_GPU HF_HOME=$HF_HOME $VLLM serve $EMBED_MODEL \
    --host 0.0.0.0 \
    --port $EMBED_PORT \
    --served-model-name $EMBED_NAME \
    --tensor-parallel-size $EMBED_TP \
    --gpu-memory-utilization 0.09 \
    --enforce-eager \
    > $EMBED_LOG 2>&1 &
EMBED_PID=$!
echo "[$(date +%T)] Embedding server PID: $EMBED_PID"

# ── 3. wait until both servers are ready ─────────────────────────────────────
wait_for_server "LLM server" $LLM_PORT
wait_for_server "Embedding server" $EMBED_PORT

# ── 4. start ALFWorld training ───────────────────────────────────────────────
echo "[$(date +%T)] Starting ALFWorld training..."
echo "[$(date +%T)] Config: $CONFIG"
echo "[$(date +%T)] Training log: $TRAIN_LOG"

cd $PROJECT
CUDA_VISIBLE_DEVICES="" $PYTHON run/run_alfworld.py --config "$CONFIG" 2>&1 | tee $TRAIN_LOG
TRAIN_EXIT=$?

# ── 5. done (cleanup via trap) ───────────────────────────────────────────────
echo "[$(date +%T)] Training finished (exit code $TRAIN_EXIT)."
