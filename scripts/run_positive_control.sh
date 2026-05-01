#!/usr/bin/env bash
# Positive control: template vs fact-based persona vectors.
# Local Qwen3-8B only. No external APIs.
#
# Time budget: ~15-20 min/task. Two tasks (LaMP-2, LaMP-7) ≈ 30-40 min.

set -e
cd "$(dirname "$0")/.."
source scripts/env.sh > /dev/null

MODEL="${MODEL:-Qwen/Qwen3-8B}"
LAYER="${LAYER:-13}"
N_USERS="${N_USERS:-30}"
ALPHA="${ALPHA:-1.0}"

echo "=== Positive Control: Template vs Fact-Based ==="
echo "  model=$MODEL  layer=$LAYER  n_users=$N_USERS  α=$ALPHA"
echo "  started: $(date -u +%FT%TZ)"

for TASK in LaMP-2 LaMP-7; do
    echo ""
    echo "----- $TASK -----"
    python -u experiments/positive_control/run_positive_control.py \
        --model "$MODEL" --task "$TASK" \
        --layer_idx "$LAYER" --n_users "$N_USERS" --alpha "$ALPHA"

    python -u experiments/positive_control/plot_comparison.py \
        --model "$(basename "$MODEL")" --task "$TASK"
done

echo ""
echo "Done: $(date -u +%FT%TZ)"
echo "Results: results/positive_control/"
echo "Figures: figures/fig_positive_control_*.pdf"
