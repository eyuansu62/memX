#!/bin/bash
# =============================================================================
# test_belief_ood.sh — OOD inference: Qwen3-30B-A3B-FP8 LLM + BeliefMemRL memory
#                      (memory checkpoint from BeliefMemRL Qwen3-4B training run)
#
# Usage:
#   bash scripts/test_belief_ood.sh [snapshot_section]
#   snapshot_section: which checkpoint to load (default: 10)
#
# Example:
#   bash scripts/test_belief_ood.sh 10
# =============================================================================
set -e

PYTHON=/home/qinbowen/miniconda3/envs/memoryrl/bin/python
VLLM=/home/qinbowen/miniconda3/envs/memoryrl/bin/vllm
PROJECT=/home/qinbowen/memX
HF_HOME=/huggingface

# ── Checkpoint ───────────────────────────────────────────────────────────────
SECTION=${1:-10}
CKPT=$PROJECT/results/alfworld/exp_alfworld_memrl_20260324-150815/local_cache/snapshot/$SECTION

# ── LLM: Qwen3-30B-A3B-FP8 (tp=2, GPU 0+1) ──────────────────────────────────
LLM_MODEL=/home/qinbowen/models/Qwen3-30B-A3B-Instruct-2507-FP8
LLM_PORT=8000
LLM_NAME=Qwen3-30B-A3B-Instruct-2507-FP8
LLM_GPU=0,1
LLM_TP=2
LLM_LOG=/tmp/vllm_qwen30b_llm.log

# ── Embedding: Qwen3-Embedding-4B (tp=1, GPU 1 — low utilization) ────────────
EMBED_MODEL=/home/qinbowen/models/Qwen3-Embedding-4B
EMBED_PORT=8001
EMBED_NAME=Qwen3-Embedding-4B
EMBED_GPU=1
EMBED_TP=1
EMBED_LOG=/tmp/vllm_embed.log

# ── Config & output ──────────────────────────────────────────────────────────
CONFIG=$PROJECT/configs/rl_alf_config.qwen30b_ood.local.yaml
INFER_LOG=$PROJECT/alfworld_belief_ood_s${SECTION}.log

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

# ── Validate checkpoint ───────────────────────────────────────────────────────
if [ ! -d "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    echo "Available snapshots:"
    ls $PROJECT/results/alfworld/exp_alfworld_memrl_20260324-150815/local_cache/snapshot/
    exit 1
fi
echo "[$(date +%T)] Using checkpoint: $CKPT"

# ── 1. Start LLM server ──────────────────────────────────────────────────────
echo "[$(date +%T)] Starting Qwen3-30B-A3B-FP8 on GPU $LLM_GPU (tp=$LLM_TP), port $LLM_PORT..."
CUDA_VISIBLE_DEVICES=$LLM_GPU HF_HOME=$HF_HOME $VLLM serve $LLM_MODEL \
    --host 0.0.0.0 \
    --port $LLM_PORT \
    --served-model-name "$LLM_NAME" \
    --tensor-parallel-size $LLM_TP \
    --gpu-memory-utilization 0.80 \
    > $LLM_LOG 2>&1 &
LLM_PID=$!
echo "[$(date +%T)] LLM server PID: $LLM_PID"

# Wait for LLM first to avoid GPU memory contention at startup
wait_for_server "Qwen3-30B LLM" $LLM_PORT

# ── 2. Start embedding server (after LLM is up) ──────────────────────────────
echo "[$(date +%T)] Starting embedding server ($EMBED_NAME) on GPU $EMBED_GPU (tp=$EMBED_TP), port $EMBED_PORT..."
CUDA_VISIBLE_DEVICES=$EMBED_GPU HF_HOME=$HF_HOME $VLLM serve $EMBED_MODEL \
    --host 0.0.0.0 \
    --port $EMBED_PORT \
    --served-model-name $EMBED_NAME \
    --tensor-parallel-size $EMBED_TP \
    --gpu-memory-utilization 0.10 \
    --max-model-len 2048 \
    --enforce-eager \
    > $EMBED_LOG 2>&1 &
EMBED_PID=$!
echo "[$(date +%T)] Embedding server PID: $EMBED_PID"

wait_for_server "Embedding" $EMBED_PORT

# ── 3. Run OOD inference ──────────────────────────────────────────────────────
echo "[$(date +%T)] Running OOD inference (Qwen3-30B-A3B + BeliefMemRL snapshot $SECTION)..."
echo "[$(date +%T)] Config:     $CONFIG"
echo "[$(date +%T)] Checkpoint: $CKPT"
echo "[$(date +%T)] Log:        $INFER_LOG"

cd $PROJECT
CUDA_VISIBLE_DEVICES="" $PYTHON run/run_alfworld.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    2>&1 | tee $INFER_LOG
EXIT_CODE=$?

echo "[$(date +%T)] Inference finished (exit code $EXIT_CODE)."
