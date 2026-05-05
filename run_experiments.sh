#!/usr/bin/env bash
# Runs all 30 experiments (3 variants × 10 seeds), optionally in parallel across GPUs.
#
# Idempotent: completed runs are recorded in runs/.done/ and skipped on re-runs.
# Delete a marker file to force a specific run to re-execute:
#   rm runs/.done/baseline_seed3
#
# Usage:
#   bash run_experiments.sh                        # single GPU, sequential
#   bash run_experiments.sh --gpus 4               # spread across 4 GPUs
#   bash run_experiments.sh --project my-project   # override W&B project name
#   bash run_experiments.sh --gpus 4 --project foo

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
N_GPUS=1
WANDB_PROJECT="LeJEPA-projector-ablation"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)    N_GPUS=$2;         shift 2 ;;
        --project) WANDB_PROJECT=$2;  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Detect whether CUDA is actually available; if not, parallelism via
# CUDA_VISIBLE_DEVICES is meaningless (MPS has a single device).
CUDA_AVAILABLE=$(uv run python -c "import torch; print(int(torch.cuda.is_available()))")
if [[ "$CUDA_AVAILABLE" != "1" && "$N_GPUS" -gt 1 ]]; then
    echo "Warning: --gpus $N_GPUS requested but CUDA is not available."
    echo "  Running sequentially on the single available device instead."
    N_GPUS=1
fi

EPOCHS=200
BS=32
LR=2e-3
LAMB=0.02
SEEDS=(0 1 2 3 4 5 6 7 8 9)
VARIANTS=("baseline" "sigreg_on_emb" "local_proj")

DONE_DIR="runs/.done"
mkdir -p "$DONE_DIR"

# ---------------------------------------------------------------------------
# Build the full list of (variant, seed) pairs
# ---------------------------------------------------------------------------
declare -a ALL_VARIANTS ALL_SEEDS
for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        ALL_VARIANTS+=("$variant")
        ALL_SEEDS+=("$seed")
    done
done
TOTAL=${#ALL_VARIANTS[@]}

# ---------------------------------------------------------------------------
# Worker: runs the subset of jobs assigned to one GPU sequentially
# ---------------------------------------------------------------------------
run_on_gpu() {
    local gpu_id=$1
    shift
    # Remaining args: interleaved variant seed variant seed ...
    while [[ $# -gt 0 ]]; do
        local variant=$1 seed=$2
        shift 2

        local marker="$DONE_DIR/${variant}_seed${seed}"
        if [[ -f "$marker" ]]; then
            echo "[GPU $gpu_id] SKIP $variant seed=$seed (already done)"
            continue
        fi

        echo "[GPU $gpu_id] START $variant seed=$seed"

        local extra_args="+V=4 +proj_dim=16"
        if [[ "$variant" == "local_proj" ]]; then
            extra_args="+V_global=2 +V_local=2"
        fi

        if [[ "$CUDA_AVAILABLE" == "1" ]]; then
            export CUDA_VISIBLE_DEVICES=$gpu_id
        fi
        uv run python train.py \
            +variant="$variant" \
            +lamb=$LAMB $extra_args +lr=$LR +bs=$BS +epochs=$EPOCHS \
            +seed="$seed" +wandb_project="$WANDB_PROJECT"

        touch "$marker"
        echo "[GPU $gpu_id] DONE  $variant seed=$seed"
    done
}

export -f run_on_gpu
export DONE_DIR LAMB LR BS EPOCHS WANDB_PROJECT CUDA_AVAILABLE

# ---------------------------------------------------------------------------
# Distribute jobs round-robin across GPUs and launch in parallel
# ---------------------------------------------------------------------------
declare -a GPU_ARGS
for (( g=0; g<N_GPUS; g++ )); do
    GPU_ARGS[$g]=""
done

for (( i=0; i<TOTAL; i++ )); do
    g=$(( i % N_GPUS ))
    GPU_ARGS[$g]+="${ALL_VARIANTS[$i]} ${ALL_SEEDS[$i]} "
done

pids=()
for (( g=0; g<N_GPUS; g++ )); do
    if [[ -n "${GPU_ARGS[$g]}" ]]; then
        # shellcheck disable=SC2086
        run_on_gpu "$g" ${GPU_ARGS[$g]} &
        pids+=($!)
    fi
done

# Wait for all GPU workers; surface any failures
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "A worker process failed (pid $pid)" >&2
        failed=1
    fi
done

if [[ $failed -eq 1 ]]; then
    echo "Some runs failed. Re-run the script to retry — completed runs will be skipped." >&2
    exit 1
fi

echo "All runs complete."
