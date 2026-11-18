import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from glmvar.evaluate import write_report
from glmvar.metrics import bootstrap_ci, choose_threshold, delong_test, point_metrics


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    good = y + rng.normal(0, 0.8, n)
    weak = y + rng.normal(0, 2.0, n)
    return y, good, weak


def test_delong_auc_matches_sklearn():
    y, good, weak = _data()
    res = delong_test(y, good, weak)
    assert np.isclose(res["auc1"], roc_auc_score(y, good))
    assert np.isclose(res["auc2"], roc_auc_score(y, weak))
    assert res["p_value"] < 1e-6  # clearly different predictors


def test_delong_identical_scores_not_significant():
    y, good, _ = _data()
    res = delong_test(y, good, good + 1e-12 * np.arange(len(good)))
    assert res["p_value"] > 0.5


def test_threshold_and_ci():
    y, good, _ = _data()
    thr = choose_threshold(y, good)
    m = point_metrics(y, good, thr)
    ci = bootstrap_ci(y, good, thr, n=200)
    for k in ("auroc", "auprc", "f1", "mcc"):
        lo, hi = ci[k]
        assert lo <= m[k] <= hi


def test_report_end_to_end(tmp_path):
    y, good, weak = _data(400)
    ids = [f"1-{i}-A-G" for i in range(len(y))]
    split = np.where(np.arange(len(y)) < 200, "val", "test")
    variants = pd.DataFrame({"variant_id": ids, "consequence": np.where(np.arange(len(y)) % 2, "missense_variant", "intron_variant")})
    vpath = tmp_path / "variants.csv"
    variants.to_csv(vpath, index=False)
    for method, s in {"cadd": weak, "dnabert2_finetune": good}.items():
        d = tmp_path / "runs" / method / "seed0"
        d.mkdir(parents=True)
        frame = pd.DataFrame({"variant_id": ids, "label": y, "score": s, "split": split})
        for sp in ("val", "test"):
            frame[frame.split == sp].to_csv(d / f"predictions_{sp}.csv", index=False)
    tables = write_report(tmp_path / "runs", vpath, tmp_path / "report", n_boot=50)
    assert tables["summary"].iloc[0]["method"] == "dnabert2_finetune"
    assert tables["delong_vs_cadd"].iloc[0]["p_value"] < 0.05
    assert set(tables["by_consequence"].group) == {"missense", "intronic"}
    assert (tmp_path / "report" / "RESULTS.md").exists()
