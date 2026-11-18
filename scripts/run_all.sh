#!/usr/bin/env bash
# Full experiment plan (Module C, Table 2). Each block answers one research question.
# Runs sequentially on one GPU; comment out blocks you have already finished.
set -euo pipefail
CFG=${CFG:-configs/default.yaml}
MODELS=(dnabert2 nt_v2_250m hyenadna_medium)
SEEDS=(0 1 2)

# 0. Data
glmvar --config "$CFG" build-dataset --all-windows

# 1. Learning-rate search (validation AUROC only; pick the best per model)
for m in "${MODELS[@]}"; do
  for lr in 1e-5 3e-5 1e-4; do
    glmvar --config "$CFG" train --model "$m" --mode finetune --seed 0 --learning-rate "$lr" --run-name "lrsearch_${m}_lr${lr}"
  done
done
python scripts/pick_learning_rates.py results/runs > results/best_lr.txt
source <(sed 's/^/export /' results/best_lr.txt)

lr_for() { local v="LR_$1"; echo "${!v}"; }

# 2. Fine-tuned models vs. each other and vs. CADD (3 seeds)
for m in "${MODELS[@]}"; do
  for s in "${SEEDS[@]}"; do
    glmvar --config "$CFG" train --model "$m" --mode finetune --seed "$s" --learning-rate "$(lr_for "$m")"
  done
done

# 3. How much does fine-tuning add? Frozen backbone + zero-shot
for m in "${MODELS[@]}"; do
  for s in "${SEEDS[@]}"; do
    glmvar --config "$CFG" train --model "$m" --mode frozen --seed "$s"
  done
done
glmvar --config "$CFG" zeroshot --model dnabert2
glmvar --config "$CFG" zeroshot --model nt_v2_250m

# 4. CADD baseline
glmvar --config "$CFG" cadd

# 5. Under what conditions? Longer context for HyenaDNA
for w in 8192 32768; do
  glmvar --config "$CFG" train --model hyenadna_medium --mode finetune --seed 0 --window "$w" --learning-rate "$(lr_for hyenadna_medium)"
done

# 6. Practical costs: data efficiency (cost/memory are logged in every summary.json)
for m in "${MODELS[@]}"; do
  for f in 0.1 0.25 0.5; do
    glmvar --config "$CFG" train --model "$m" --mode finetune --seed 0 --train-fraction "$f" --learning-rate "$(lr_for "$m")"
  done
done

# 7. Tables and figures
mkdir -p results/lrsearch && mv results/runs/lrsearch_* results/lrsearch/ 2>/dev/null || true
glmvar --config "$CFG" report
