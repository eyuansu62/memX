#!/bin/bash
# =============================================================================
# run_diagnostic.sh — Run MemRL vs BeliefMemRL diagnostic experiments
#                     (5 sections, 20% data) and produce analysis outputs.
#
# Usage:
#   bash scripts/run_diagnostic.sh
# =============================================================================
set -e

PYTHON=/home/qinbowen/miniconda3/envs/memoryrl/bin/python
VLLM=/home/qinbowen/miniconda3/envs/memoryrl/bin/vllm
PROJECT=/home/qinbowen/memX
HF_HOME=/huggingface

CONFIG=$PROJECT/configs/rl_alf_config.diagnostic.local.yaml
MEMRL_LOG=$PROJECT/logs/memrl_alfworld.jsonl
BELIEF_LOG=$PROJECT/logs/belief_memrl_alfworld.jsonl
OUT_DIR=$PROJECT/results/diagnostics
TRAIN_LOG_BASE=$PROJECT/diagnostic_train

# ── LLM server ───────────────────────────────────────────────────────────────
LLM_MODEL=/home/qinbowen/models/Qwen3-4B-Instruct-2507
LLM_PORT=8000
LLM_NAME=Qwen3-4B-Instruct-2507
LLM_GPU=0,1
LLM_TP=2
LLM_LOG=/tmp/vllm_llm.log

# ── Embedding server ──────────────────────────────────────────────────────────
EMBED_MODEL=/home/qinbowen/models/Qwen3-Embedding-4B
EMBED_PORT=8001
EMBED_NAME=Qwen3-Embedding-4B
EMBED_GPU=1
EMBED_TP=1
EMBED_LOG=/tmp/vllm_embed.log

# ── helpers ───────────────────────────────────────────────────────────────────
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

# ── 1. Start servers ──────────────────────────────────────────────────────────
echo "[$(date +%T)] Starting LLM server ($LLM_NAME) on GPU $LLM_GPU (tp=$LLM_TP)..."
CUDA_VISIBLE_DEVICES=$LLM_GPU HF_HOME=$HF_HOME $VLLM serve $LLM_MODEL \
    --host 0.0.0.0 --port $LLM_PORT \
    --served-model-name $LLM_NAME \
    --tensor-parallel-size $LLM_TP \
    --gpu-memory-utilization 0.75 \
    > $LLM_LOG 2>&1 &
LLM_PID=$!

echo "[$(date +%T)] Starting embedding server ($EMBED_NAME) on GPU $EMBED_GPU..."
CUDA_VISIBLE_DEVICES=$EMBED_GPU HF_HOME=$HF_HOME $VLLM serve $EMBED_MODEL \
    --host 0.0.0.0 --port $EMBED_PORT \
    --served-model-name $EMBED_NAME \
    --tensor-parallel-size $EMBED_TP \
    --gpu-memory-utilization 0.15 \
    --max-model-len 4096 \
    --enforce-eager \
    > $EMBED_LOG 2>&1 &
EMBED_PID=$!

wait_for_server "LLM" $LLM_PORT
wait_for_server "Embedding" $EMBED_PORT

cd $PROJECT

# ── 2. Run MemRL baseline (original service, with logging) ────────────────────
echo "[$(date +%T)] === Run 1/2: MemRL baseline ==="
rm -f $MEMRL_LOG
rm -rf .memos results/mem_cubes
CUDA_VISIBLE_DEVICES="" $PYTHON run/run_alfworld.py \
    --config "$CONFIG" \
    --memory_service original \
    --log_path "$MEMRL_LOG" \
    2>&1 | tee ${TRAIN_LOG_BASE}_memrl.log

echo "[$(date +%T)] MemRL run complete. Events: $(wc -l < $MEMRL_LOG)"

# ── 3. Run BeliefMemRL (with logging) ─────────────────────────────────────────
echo "[$(date +%T)] === Run 2/2: BeliefMemRL ==="
rm -f $BELIEF_LOG
rm -rf .memos results/mem_cubes
CUDA_VISIBLE_DEVICES="" $PYTHON run/run_alfworld.py \
    --config "$CONFIG" \
    --memory_service belief \
    --log_path "$BELIEF_LOG" \
    2>&1 | tee ${TRAIN_LOG_BASE}_belief.log

echo "[$(date +%T)] BeliefMemRL run complete. Events: $(wc -l < $BELIEF_LOG)"

# ── 4. Run analysis scripts ───────────────────────────────────────────────────
echo "[$(date +%T)] === Running diagnostic analyses ==="

$PYTHON -m memrl.analysis.cross_epoch_retrieval \
    --memrl "$MEMRL_LOG" --belief "$BELIEF_LOG" --out_dir "$OUT_DIR"

$PYTHON -m memrl.analysis.memory_survival \
    --memrl "$MEMRL_LOG" --belief "$BELIEF_LOG" --out_dir "$OUT_DIR"

$PYTHON -m memrl.analysis.posterior_calibration \
    --belief "$BELIEF_LOG" --out_dir "$OUT_DIR"

$PYTHON -m memrl.analysis.summary_table \
    --memrl "$MEMRL_LOG" --belief "$BELIEF_LOG" --out_dir "$OUT_DIR"

echo "[$(date +%T)] All diagnostics complete. Results in $OUT_DIR"
