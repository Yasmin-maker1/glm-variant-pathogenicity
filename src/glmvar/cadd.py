"""CADD v1.7 baseline (Section III.E): look up precomputed PHRED scores.

Two options:
  1. ``lookup_tabix`` on the full ``whole_genome_SNVs.tsv.gz`` (+ .tbi) from
     https://cadd.bihealth.org/download (~80 GB; needs pysam).
  2. ``lookup_table`` on a smaller TSV you scored with the CADD web service
     (upload ``data/processed/variants.vcf`` and download the result).
CADD is not retrained; only its decision threshold is chosen on validation data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def lookup_tabix(df: pd.DataFrame, tsv_gz: str | Path, column: str = "PHRED") -> np.ndarray:
    import pysam

    tbx = pysam.TabixFile(str(tsv_gz))
    header = None
    for line in tbx.header:
        if line.startswith("#Chrom") or line.startswith("#CHROM"):
            header = line.lstrip("#").split("\t")
    col = header.index(column) if header else 5
    scores = np.full(len(df), np.nan)
    for i, r in enumerate(df.itertuples(index=False)):
        for row in tbx.fetch(str(r.chrom), r.pos - 1, r.pos):
            f = row.split("\t")
            if int(f[1]) == r.pos and f[2] == r.ref and f[3] == r.alt:
                scores[i] = float(f[col])
                break
    return scores


def lookup_table(df: pd.DataFrame, table: str | Path, column: str = "PHRED") -> np.ndarray:
    cadd = pd.read_csv(table, sep="\t", comment=None, skiprows=lambda i: i == 0 and _is_version_line(table), dtype={"#Chrom": str, "Chrom": str})
    cadd = cadd.rename(columns={"#Chrom": "Chrom"})
    cadd["variant_id"] = cadd["Chrom"].astype(str).str.removeprefix("chr") + "-" + cadd["Pos"].astype(str) + "-" + cadd["Ref"] + "-" + cadd["Alt"]
    m = df[["variant_id"]].merge(cadd[["variant_id", column]].drop_duplicates("variant_id"), on="variant_id", how="left")
    return m[column].to_numpy(dtype=float)


def _is_version_line(path) -> bool:
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return fh.readline().startswith("## CADD")
