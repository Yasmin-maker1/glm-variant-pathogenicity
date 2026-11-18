"""Evaluation metrics (Section III.F).

AUROC (primary), AUPRC, F1 and MCC at a validation-chosen threshold,
bootstrap 95% confidence intervals, and DeLong's test for paired AUROCs.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score


def choose_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """Threshold that maximizes F1 on the validation set (used for every method, incl. CADD)."""
    p, r, t = precision_recall_curve(y_val, s_val)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)[:-1]
    return float(t[int(np.argmax(f1))]) if len(t) else 0.5


def point_metrics(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    pred = (s >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y, s),
        "auprc": average_precision_score(y, s),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
    }


def bootstrap_ci(y: np.ndarray, s: np.ndarray, threshold: float, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    draws: dict[str, list[float]] = {k: [] for k in ("auroc", "auprc", "f1", "mcc")}
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        for k, v in point_metrics(y[b], s[b], threshold).items():
            draws[k].append(v)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in draws.items() if v}


# ---- DeLong's test (fast implementation of Sun & Xu, 2014) -------------------

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """preds_sorted: [k, n] with the m positives first."""
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """Two-sided DeLong test for H0: AUROC(s1) == AUROC(s2) on the same samples."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="stable")  # positives first
    m = int(y.sum())
    preds = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff), "z": float(z), "p_value": float(p)}
