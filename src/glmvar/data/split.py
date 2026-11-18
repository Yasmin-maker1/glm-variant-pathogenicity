"""Gene-level train/validation/test split (Section III.C) and learning-curve subsets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def gene_split(
    df: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 13,
    gene_col: str = "gene",
) -> pd.DataFrame:
    """Assign every variant of a gene to the same partition.

    Genes are shuffled with a fixed seed and assigned greedily so that each
    partition receives approximately the requested fraction of *variants*.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    counts = df.groupby(gene_col).size()
    genes = counts.index.to_numpy()
    rng.shuffle(genes)
    total = counts.sum()
    targets = np.cumsum(fractions) * total
    names = ["train", "val", "test"]
    assignment, running, idx = {}, 0, 0
    for g in genes:
        while idx < 2 and running >= targets[idx]:
            idx += 1
        assignment[g] = names[idx]
        running += counts[g]
    out = df.copy()
    out["split"] = out[gene_col].map(assignment)
    return out


def check_no_leakage(df: pd.DataFrame, gene_col: str = "gene") -> None:
    per_gene = df.groupby(gene_col)["split"].nunique()
    leaked = per_gene[per_gene > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} genes appear in more than one split")


def subsample_train_genes(df: pd.DataFrame, fraction: float, seed: int = 0, gene_col: str = "gene") -> pd.DataFrame:
    """Keep ``fraction`` of the *training genes* (val/test untouched) for learning curves."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1:
        return df
    train_genes = np.array(sorted(df.loc[df.split == "train", gene_col].unique()))
    rng = np.random.default_rng(seed)
    n = max(1, int(round(fraction * len(train_genes))))
    keep = set(rng.choice(train_genes, size=n, replace=False))
    mask = (df.split != "train") | df[gene_col].isin(keep)
    return df.loc[mask].reset_index(drop=True)
