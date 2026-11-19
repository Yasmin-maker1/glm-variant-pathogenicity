"""CADD v1.7 baseline (Section III.E): look up precomputed PHRED scores.

Three options:
  1. ``lookup_tabix`` on the full ``whole_genome_SNVs.tsv.gz`` (+ .tbi). The path
     may be a local file (~80 GB download) **or the https URL** of the file on
     the CADD server: htslib then fetches only the small blocks it needs.
  2. ``lookup_table`` on a smaller TSV you scored with the CADD web service
     (upload ``data/processed/variants.vcf`` and download the result).
CADD is not retrained; only its decision threshold is chosen on validation data.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

CADD_URL = "https://kircherlab.bihealth.org/download/CADD/v1.7/GRCh38/whole_genome_SNVs.tsv.gz"


def lookup_tabix(df: pd.DataFrame, tsv_gz: str | Path, column: str = "PHRED", threads: int = 8) -> np.ndarray:
    import pysam

    local = threading.local()

    def handle():
        if not hasattr(local, "tbx"):
            local.tbx = pysam.TabixFile(str(tsv_gz))
        return local.tbx

    tbx0 = handle()
    header = None
    for line in tbx0.header:
        if line.startswith("#Chrom") or line.startswith("#CHROM"):
            header = line.lstrip("#").split("\t")
    col = header.index(column) if header else 5

    rows = list(df[["chrom", "pos", "ref", "alt"]].itertuples(index=False))

    def one(r):
        for attempt in range(3):
            try:
                for row in handle().fetch(str(r.chrom), r.pos - 1, r.pos):
                    f = row.split("\t")
                    if int(f[1]) == r.pos and f[2] == r.ref and f[3] == r.alt:
                        return float(f[col])
                return np.nan
            except (OSError, ValueError):
                if attempt == 2:
                    return np.nan
                local.__dict__.pop("tbx", None)  # reopen after a network hiccup
        return np.nan

    with ThreadPoolExecutor(max_workers=threads) as pool:
        return np.array(list(pool.map(one, rows)), dtype=float)


def lookup_table(df: pd.DataFrame, table: str | Path, column: str = "PHRED") -> np.ndarray:
    cadd = pd.read_csv(table, sep="\t", skiprows=lambda i: i == 0 and _is_version_line(table), dtype={"#Chrom": str, "Chrom": str})
    cadd = cadd.rename(columns={"#Chrom": "Chrom"})
    cadd["variant_id"] = cadd["Chrom"].astype(str).str.removeprefix("chr") + "-" + cadd["Pos"].astype(str) + "-" + cadd["Ref"] + "-" + cadd["Alt"]
    m = df[["variant_id"]].merge(cadd[["variant_id", column]].drop_duplicates("variant_id"), on="variant_id", how="left")
    return m[column].to_numpy(dtype=float)


def _is_version_line(path) -> bool:
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return fh.readline().startswith("## CADD")
