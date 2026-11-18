# Fine-Tuning Genomic Language Models for Variant Pathogenicity

**Do modern DNA language models beat a mature, feature-engineered tool at telling pathogenic from benign variants, and under what conditions?**

This repository contains the full experimental pipeline for my CSCI 7610 Graduate Technical Seminar project (Georgia Southern University). It fine-tunes three publicly available genomic language models on ClinVar single-nucleotide variants and compares them, under academic-scale compute, with each other and with **CADD v1.7**.

| Model | Checkpoint | Architecture | Tokens | Fine-tuning |
|---|---|---|---|---|
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` | Transformer encoder | BPE | LoRA (r=8, α=16) |
| Nucleotide Transformer v2 | `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` | Transformer encoder | 6-mers | LoRA (r=8, α=16) |
| HyenaDNA | `LongSafari/hyenadna-medium-160k-seqlen-hf` | Hyena long convolutions | single nt | Full |
| **Baseline:** CADD v1.7 | precomputed PHRED scores | Feature-engineered | – | not retrained |

> **Status:** pipeline implemented and unit-tested; experiments on the full ClinVar dataset are in progress. Results will appear in [`results/`](results/).

---

## Method at a glance

```
ClinVar (GRCh38, fixed release)
  └─ SNVs, ≥1 review star, P/LP = 1 vs B/LB = 0 (VUS & conflicting removed)
      └─ 1,024-bp reference window + alternate window (variant at center)
          └─ split BY GENE: 80 / 10 / 10  (no gene in two splits)
              ├─ DNABERT-2  ─┐
              ├─ NT v2 250M ─┼─ shared ref/alt pair classifier → P(pathogenic)
              ├─ HyenaDNA   ─┘
              └─ CADD v1.7 PHRED (lookup)
                  └─ same test set: AUROC, AUPRC, F1, MCC, bootstrap 95% CI, DeLong vs CADD
```

**Pair classifier.** The reference and alternate windows pass through the same backbone. Token embeddings within ±32 bp of the variant are averaged, and `[ref, alt, alt − ref]` goes to a two-layer head. Training uses class-weighted BCE, AdamW, mixed precision, an effective batch of 32, up to 10 epochs and early stopping (patience 2) on validation AUROC.

**Research questions → experiments** (Module C, Table 2)

| Question | Experiment | Measurement |
|---|---|---|
| Do fine-tuned models beat CADD? | fine-tune each model; score test set with CADD | AUROC (primary), AUPRC, F1, MCC + 95% CI; DeLong's test |
| How do the models compare? | same data, window, head and budget; 3 seeds | same metrics, mean ± SD |
| Under what conditions? | per-consequence evaluation; HyenaDNA with 8 kb / 32 kb windows | metrics per consequence type and window |
| How much does fine-tuning add? | frozen-backbone head; zero-shot masked LLR | ΔAUROC fine-tuned vs frozen vs zero-shot |
| What does it cost? | resource logging; 10/25/50/100% of training genes | GPU-hours, peak memory, variants/s, learning curves |

---

## Repository layout

```
configs/            default.yaml (full study), smoke.yaml (quick check)
scripts/            download_data.sh, run_all.sh, pick_learning_rates.py
src/glmvar/
  data/             clinvar.py · sequences.py · split.py · vep.py
  models/           registry.py · tokens.py · classifier.py · loading.py
  train.py          fine-tuning / frozen training loop
  zeroshot.py       masked-LM log-likelihood-ratio scores
  cadd.py           CADD v1.7 lookup
  metrics.py        AUROC/AUPRC/F1/MCC, bootstrap CI, DeLong test
  evaluate.py       results tables;  plots.py  ROC + learning curves
  cli.py            `glmvar` command
tests/              offline unit tests (tiny genome, toy encoder)
docs/               methodology notes
results/            tables and figures (generated)
```

## Quick start

```bash
git clone https://github.com/<your-user>/glm-variant-pathogenicity.git
cd glm-variant-pathogenicity
python -m venv .venv && source .venv/bin/activate
pip install -e ".[bio,dev]"
pytest -q                                   # offline tests, CPU only
```

A GPU with 16–24 GB (for example an A10, L4, RTX 4090, or Colab/Kaggle T4 for small runs) is enough for every experiment.

### 1. Data

```bash
bash scripts/download_data.sh               # ClinVar + GRCh38 (+ CADD with DOWNLOAD_CADD=1)
glmvar build-dataset --all-windows          # → data/processed/variants_w1024.parquet (+ 8 kb, 32 kb)
```

### 2. One run

```bash
glmvar train --model dnabert2 --mode finetune --seed 0
glmvar train --model nt_v2_250m --mode frozen --seed 0
glmvar zeroshot --model nt_v2_250m
glmvar cadd
glmvar report                               # → results/report/RESULTS.md, roc_test.png, learning_curves.png
```

Smoke test with tiny settings: `glmvar --config configs/smoke.yaml train --model hyenadna_medium --limit 200`.

### 3. Full study

```bash
bash scripts/run_all.sh
```

### CADD baseline without the 80 GB file

Run `glmvar export-vcf`, upload `data/processed/variants.vcf` to the [CADD web service](https://cadd.bihealth.org/score) (v1.7, GRCh38), download the result and set `cadd.scored_table` in the config.

## Reproducibility

Every run writes `results/runs/<method>/seed<k>/` with `predictions_val.csv`, `predictions_test.csv` and `summary.json` (seed, epochs, validation AUROC, GPU-hours, peak GPU memory, variants per second, trainable parameters). Record the ClinVar release date and the commit hash with your results.

## Limitations

ClinVar labels contain errors and favor well-studied genes and populations. Only SNVs are studied (no indels or structural variants). The common 1,024-bp window and modest budget may understate what each model can do with longer context or larger checkpoints. This is deliberate: the goal is a comparison under realistic academic conditions.

## Key references

- Zhou, Z. et al. (2024). DNABERT-2. *ICLR 2024*.
- Dalla-Torre, H. et al. (2025). Nucleotide Transformer. *Nature Methods*, 22, 287–297.
- Nguyen, E. et al. (2023). HyenaDNA. *NeurIPS 2023*.
- Schubach, M. et al. (2024). CADD v1.7. *Nucleic Acids Research*.
- Kircher, M. et al. (2014). CADD. *Nature Genetics*.
- Landrum, M. J. et al. (2018). ClinVar. *Nucleic Acids Research*, 46(D1).
- Richards, S. et al. (2015). ACMG/AMP guidelines. *Genetics in Medicine*, 17(5).
- Marin, F. I. et al. (2024). BEND. *ICLR 2024*.
- Hu, E. J. et al. (2022). LoRA. *ICLR 2022*.
- DeLong, E. R. et al. (1988). *Biometrics*, 44(3).

## Author

Yasmin Orduz · CSCI 7610 Graduate Technical Seminar · Georgia Southern University · Advisor: Dr. Ray Hashemi

MIT License.
