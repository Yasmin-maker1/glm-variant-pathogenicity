"""Build the results tables from saved prediction files.

Every method (fine-tuned / frozen / zero-shot model, or CADD) writes
``predictions_val.csv`` and ``predictions_test.csv`` with columns
``variant_id,label,score,split`` into ``results/runs/<method>/seed<k>/``.
This module joins them with the variant table and reports, on the test set:
  * overall AUROC/AUPRC/F1/MCC with bootstrap 95% CIs
  * DeLong's test of each method against CADD
  * metrics per consequence group and on consequence-matched subsets
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data.clinvar import consequence_group
from .metrics import bootstrap_ci, choose_threshold, delong_test, point_metrics


def load_runs(runs_dir: str | Path) -> dict[str, list[tuple[pd.DataFrame, pd.DataFrame]]]:
    runs: dict[str, list] = {}
    for method_dir in sorted(Path(runs_dir).iterdir()):
        if not method_dir.is_dir():
            continue
        for seed_dir in sorted(method_dir.glob("seed*")):
            v, t = seed_dir / "predictions_val.csv", seed_dir / "predictions_test.csv"
            if v.exists() and t.exists():
                runs.setdefault(method_dir.name, []).append((pd.read_csv(v), pd.read_csv(t)))
    return runs


def evaluate_method(val: pd.DataFrame, test: pd.DataFrame, n_boot: int = 1000) -> dict:
    val, test = val.dropna(subset=["score"]), test.dropna(subset=["score"])
    thr = choose_threshold(val.label.to_numpy(), val.score.to_numpy())
    y, s = test.label.to_numpy(), test.score.to_numpy()
    res = point_metrics(y, s, thr)
    ci = bootstrap_ci(y, s, thr, n=n_boot)
    out = {"threshold": thr, "n_test": len(test)}
    for k, v in res.items():
        out[k] = v
        out[f"{k}_ci_low"], out[f"{k}_ci_high"] = ci.get(k, (np.nan, np.nan))
    return out


def summary_table(runs: dict, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for method, seeds in runs.items():
        per_seed = [evaluate_method(v, t, n_boot) for v, t in seeds]
        df = pd.DataFrame(per_seed)
        row = {"method": method, "n_seeds": len(per_seed)}
        for k in ("auroc", "auprc", "f1", "mcc"):
            row[k] = df[k].mean()
            row[f"{k}_sd"] = df[k].std(ddof=1) if len(df) > 1 else 0.0
            row[f"{k}_ci"] = f"[{df[f'{k}_ci_low'].mean():.3f}, {df[f'{k}_ci_high'].mean():.3f}]"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def mean_test_scores(seeds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    tests = [t[["variant_id", "label", "score"]] for _, t in seeds]
    merged = tests[0].rename(columns={"score": "s0"})
    for i, t in enumerate(tests[1:], 1):
        merged = merged.merge(t[["variant_id", "score"]].rename(columns={"score": f"s{i}"}), on="variant_id")
    merged["score"] = merged[[c for c in merged if c.startswith("s") and c[1:].isdigit()]].mean(axis=1)
    return merged[["variant_id", "label", "score"]]


def delong_vs_baseline(runs: dict, baseline: str = "cadd") -> pd.DataFrame:
    if baseline not in runs:
        return pd.DataFrame()
    base = mean_test_scores(runs[baseline])
    rows = []
    for method, seeds in runs.items():
        if method == baseline:
            continue
        m = mean_test_scores(seeds).merge(base, on=["variant_id", "label"], suffixes=("", "_base")).dropna()
        res = delong_test(m.label.to_numpy(), m.score.to_numpy(), m.score_base.to_numpy())
        rows.append({"method": method, "vs": baseline, "n": len(m), **res})
    return pd.DataFrame(rows)


def by_consequence(runs: dict, variants: pd.DataFrame, min_n: int = 50) -> pd.DataFrame:
    groups = variants[["variant_id", "consequence"]].copy()
    groups["group"] = groups.consequence.map(consequence_group)
    rows = []
    for method, seeds in runs.items():
        test = mean_test_scores(seeds).merge(groups, on="variant_id").dropna(subset=["score"])
        for g, sub in test.groupby("group"):
            if len(sub) < min_n or sub.label.nunique() < 2:
                continue
            rows.append({"method": method, "group": g, "n": len(sub), "pos_frac": sub.label.mean(),
                         "auroc": point_metrics(sub.label.to_numpy(), sub.score.to_numpy(), 0.5)["auroc"]})
    return pd.DataFrame(rows)


def write_report(runs_dir: str | Path, variants_path: str | Path, out_dir: str | Path, n_boot: int = 1000) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_dir)
    variants = pd.read_parquet(variants_path) if str(variants_path).endswith(".parquet") else pd.read_csv(variants_path)
    tables = {
        "summary": summary_table(runs, n_boot),
        "delong_vs_cadd": delong_vs_baseline(runs, "cadd"),
        "by_consequence": by_consequence(runs, variants),
    }
    md = ["# Results\n"]
    for name, tab in tables.items():
        tab.to_csv(out_dir / f"{name}.csv", index=False)
        md.append(f"\n## {name}\n\n" + (tab.round(4).to_markdown(index=False) if len(tab) else "_no runs yet_") + "\n")
    (out_dir / "RESULTS.md").write_text("".join(md))
    return tables
