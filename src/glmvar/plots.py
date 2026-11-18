"""Figures for the report: ROC curves and data-efficiency learning curves."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .evaluate import load_runs, mean_test_scores


def plot_roc(runs: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for method, seeds in runs.items():
        if "_frac" in method or "_w" in method:
            continue
        t = mean_test_scores(seeds).dropna()
        fpr, tpr, _ = roc_curve(t.label, t.score)
        ax.plot(fpr, tpr, lw=1.6, label=f"{method} ({roc_auc_score(t.label, t.score):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test-set ROC")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_learning_curves(runs_dir: Path, out: Path) -> None:
    rows = []
    for summ in Path(runs_dir).glob("*/seed*/summary.json"):
        s = json.loads(summ.read_text())
        name = summ.parent.parent.name
        if s.get("mode") != "finetune":
            continue
        frac = float(re.search(r"_frac([0-9.]+)", name).group(1)) if "_frac" in name else 1.0
        test = pd.read_csv(summ.parent / "predictions_test.csv")
        rows.append({"model": s["model"], "fraction": frac, "auroc": roc_auc_score(test.label, test.score)})
    if not rows:
        return
    df = pd.DataFrame(rows).groupby(["model", "fraction"]).auroc.mean().reset_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    for m, g in df.groupby("model"):
        ax.plot(g.fraction * 100, g.auroc, marker="o", label=m)
    ax.set(xlabel="% of training genes", ylabel="Test AUROC", title="Data efficiency")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_all(runs_dir, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_dir)
    if runs:
        plot_roc(runs, out_dir / "roc_test.png")
    plot_learning_curves(Path(runs_dir), out_dir / "learning_curves.png")
