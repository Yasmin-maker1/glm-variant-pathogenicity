"""Optional consequence annotation with the Ensembl Variant Effect Predictor (VEP).

The ClinVar ``MC`` field already gives a molecular consequence for most variants,
and ``load_clinvar`` uses it by default. To follow Module C exactly (VEP
annotation), write the variants to a VCF, run VEP, and merge the result back.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


def write_vcf(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for r in df.sort_values(["chrom", "pos"]).itertuples():
            fh.write(f"{r.chrom}\t{r.pos}\t{r.variant_id}\t{r.ref}\t{r.alt}\t.\t.\t.\n")
    return path


def run_vep(vcf: str | Path, out_tsv: str | Path, cache_dir: str | Path, fork: int = 4) -> Path:
    """Run VEP (must be installed, e.g. ``conda install -c bioconda ensembl-vep``)."""
    cmd = [
        "vep", "--input_file", str(vcf), "--output_file", str(out_tsv),
        "--tab", "--offline", "--cache", "--dir_cache", str(cache_dir),
        "--assembly", "GRCh38", "--pick", "--fields", "Uploaded_variation,Consequence,SYMBOL",
        "--fork", str(fork), "--force_overwrite", "--no_stats",
    ]
    subprocess.run(cmd, check=True)
    return Path(out_tsv)


def read_vep_tab(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.lstrip("#").rstrip("\n").split("\t")
                continue
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    out = pd.DataFrame(rows)
    out["consequence"] = out["Consequence"].str.split(",").str[0]
    return out.rename(columns={"Uploaded_variation": "variant_id"})[["variant_id", "consequence"]]


def merge_vep(df: pd.DataFrame, vep: pd.DataFrame) -> pd.DataFrame:
    merged = df.drop(columns=["consequence"]).merge(vep, on="variant_id", how="left")
    merged["consequence"] = merged["consequence"].fillna("unknown")
    return merged
