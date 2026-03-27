#!/bin/bash
# =============================================================================
# test_qwen30b_ood.sh — Qwen3-30B-A3B-FP8 OOD inference with Qwen3-4B memory
# =============================================================================
set -e

PYTHON=/home/qinbowen/miniconda3/envs/memrl/bin/python
VLLM=/home/qinbowen/miniconda3/envs/dschat/bin/vllm
PROJECT=/home/qinbowen/MemRL
HF_HOME=/huggingface

# ── LLM: Qwen3-30B-A3B-FP8 (tp=2, GPU 0+1) ───────────────────────────────
LLM_MODEL=/home/qinbowen/models/Qwen3-30B-A3B-Instruct-2507-FP8
LLM_PORT=8000
LLM_NAME=Qwen3-30B-A3B-Instruct-2507-FP8
LLM_GPU=0,1
LLM_TP=2
LLM_LOG=/tmp/vllm_qwen30b_llm.log

# ── Embedding: Qwen3-Embedding-4B (tp=1, GPU 2) ───────────────────────────
EMBED_MODEL=/home/qinbowen/models/Qwen3-Embedding-4B
EMBED_PORT=8001
EMBED_NAME=Qwen3-Embedding-4B
EMBED_GPU=2
EMBED_TP=1
EMBED_LOG=/tmp/vllm_embed.log

# ── Inference ───────────────────────────────────────────────────────────────
CONFIG=$PROJECT/configs/rl_alf_config.qwen30b_ood.yaml
CKPT=$PROJECT/results/alfworld/exp_alfworld_qwen3_4b_20260319-082912/local_cache/snapshot/8
INFER_LOG=$PROJECT/alfworld_qwen30b_ood.log

# ── helpers ──────────────────────────────────────────────────────────────────
wait_for_server() {
    local name=$1 port=$2
    echo "[$(date +%T)] Waiting for $name on port $port..."
    for i in $(seq 1 120); do
        if curl -sf http://localhost:$port/health > /dev/null 2>&1; then
            echo "[$(date +%T)] $name is ready."
            return 0
        fi
        sleep 5
    done
    echo "[$(date +%T)] ERROR: $name did not start within 10 minutes."
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
echo "[$(date +%T)] Starting Qwen3-30B-A3B-FP8 on GPU $LLM_GPU (tp=$LLM_TP), port $LLM_PORT..."
CUDA_VISIBLE_DEVICES=$LLM_GPU HF_HOME=$HF_HOME $VLLM serve $LLM_MODEL \
    --host 0.0.0.0 \
    --port $LLM_PORT \
    --served-model-name "$LLM_NAME" \
    --tensor-parallel-size $LLM_TP \
    --gpu-memory-utilization 0.85 \
    > $LLM_LOG 2>&1 &
LLM_PID=$!

# ── 2. wait for LLM first, then start embedding ─────────────────────────────
wait_for_server "Qwen3-30B LLM" $LLM_PORT

# Start embedding after LLM is ready to avoid GPU memory contention
echo "[$(date +%T)] Starting embedding server ($EMBED_NAME) on GPU $EMBED_GPU (tp=$EMBED_TP), port $EMBED_PORT..."
CUDA_VISIBLE_DEVICES=$EMBED_GPU HF_HOME=$HF_HOME $VLLM serve $EMBED_MODEL \
    --host 0.0.0.0 \
    --port $EMBED_PORT \
    --served-model-name $EMBED_NAME \
    --tensor-parallel-size $EMBED_TP \
    --gpu-memory-utilization 0.12 \
    --enforce-eager \
    --max-model-len 8192 \
    > $EMBED_LOG 2>&1 &
EMBED_PID=$!

wait_for_server "Embedding" $EMBED_PORT

# ── 4. run inference ─────────────────────────────────────────────────────────
echo "[$(date +%T)] Running OOD inference (Qwen3-30B-A3B + Qwen3-4B memory)..."
echo "[$(date +%T)] Config: $CONFIG"
echo "[$(date +%T)] Checkpoint: $CKPT"

cd $PROJECT
CUDA_VISIBLE_DEVICES="" $PYTHON run/run_alfworld.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    2>&1 | tee $INFER_LOG
EXIT_CODE=$?

echo "[$(date +%T)] Inference finished (exit code $EXIT_CODE)."
