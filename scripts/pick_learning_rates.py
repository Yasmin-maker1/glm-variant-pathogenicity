"""Print LR_<model>=<best lr> lines from the learning-rate search runs."""

import json
import re
import sys
from pathlib import Path

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "results/runs")
best: dict[str, tuple[float, str]] = {}
for summ in runs.glob("lrsearch_*/seed0/summary.json"):
    m = re.match(r"lrsearch_(.+)_lr(.+)", summ.parent.parent.name)
    model, lr = m.group(1), m.group(2)
    auc = json.loads(summ.read_text())["best_val_auroc"]
    if model not in best or auc > best[model][0]:
        best[model] = (auc, lr)
for model, (auc, lr) in sorted(best.items()):
    print(f"LR_{model}={lr}")
