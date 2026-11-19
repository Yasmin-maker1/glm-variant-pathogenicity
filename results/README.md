# Results

## First, reduced run (November 2026)

- ClinVar GRCh38 release 2026-11-15; SNVs with at least one review star; split by gene 80/10/10, then a random subset of each split (24,000 train / 3,000 validation / 4,000 test variants; 10.7% of test variants pathogenic).
- One seed, fixed learning rate (1e-4), up to 3 epochs, one free Google Colab T4 GPU (`configs/colab.yaml`).

Files in `report/`:

- `RESULTS.md` - summary table, DeLong's test vs CADD, per-consequence AUROC
- `summary.csv`, `delong_vs_cadd.csv`, `by_consequence.csv`, `cost.csv`
- `roc_test.png` - test-set ROC curves

Per-run predictions and summaries are kept in `runs/` (not committed).

