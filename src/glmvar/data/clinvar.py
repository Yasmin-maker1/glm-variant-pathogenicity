"""Parse a ClinVar GRCh38 VCF into a labeled table of single-nucleotide variants.

Labeling follows Module C, Section III.A:
  * positive (1): Pathogenic, Likely pathogenic, Pathogenic/Likely pathogenic
  * negative (0): Benign, Likely benign, Benign/Likely benign
  * excluded: uncertain significance, conflicting interpretations, everything else
  * only variants with review status of at least one star are kept
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator

import pandas as pd

PATHOGENIC = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}

# ClinVar CLNREVSTAT value -> star rating.
REVIEW_STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_conflicting_interpretations": 1,
    "no_assertion_criteria_provided": 0,
    "no_classification_provided": 0,
    "no_assertion_provided": 0,
    "no_interpretation_for_the_single_variant": 0,
    "no_classifications_from_unflagged_records": 0,
}

VALID_CHROMS = {str(c) for c in range(1, 23)} | {"X", "Y", "MT"}
BASES = set("ACGT")


def _open(path: str | Path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in info.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
        else:
            out[item] = "1"
    return out


def primary_significance(clnsig: str) -> str:
    """'Pathogenic|risk_factor' or 'Pathogenic,_low_penetrance' -> 'Pathogenic'."""
    return clnsig.split("|")[0].split(",")[0]


def label_from_clnsig(clnsig: str) -> int | None:
    sig = primary_significance(clnsig)
    if sig in PATHOGENIC:
        return 1
    if sig in BENIGN:
        return 0
    return None


def parse_molecular_consequence(mc: str | None) -> str:
    """ClinVar MC field 'SO:0001583|missense_variant,SO:...' -> 'missense_variant'."""
    if not mc:
        return "unknown"
    first = mc.split(",")[0]
    return first.split("|")[1] if "|" in first else first


def iter_vcf_records(path: str | Path) -> Iterator[dict]:
    with _open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            chrom, pos, vid, ref, alt, _qual, _filt, info = line.rstrip("\n").split("\t")[:8]
            yield {"chrom": chrom, "pos": int(pos), "id": vid, "ref": ref, "alt": alt, "info": _parse_info(info)}


def load_clinvar(
    vcf_path: str | Path,
    min_stars: int = 1,
    drop_multi_gene: bool = True,
) -> pd.DataFrame:
    """Return one row per retained SNV with columns used downstream."""
    rows = []
    for rec in iter_vcf_records(vcf_path):
        info = rec["info"]
        chrom = rec["chrom"].removeprefix("chr")
        if chrom not in VALID_CHROMS:
            continue
        ref, alt = rec["ref"].upper(), rec["alt"].upper()
        if len(ref) != 1 or len(alt) != 1 or ref not in BASES or alt not in BASES:
            continue  # SNVs only
        label = label_from_clnsig(info.get("CLNSIG", ""))
        if label is None:
            continue
        stars = REVIEW_STARS.get(info.get("CLNREVSTAT", ""), 0)
        if stars < min_stars:
            continue
        genes = [g.split(":")[0] for g in info.get("GENEINFO", "").split("|") if g]
        if not genes or (drop_multi_gene and len(genes) > 1):
            continue
        rows.append(
            {
                "variant_id": f"{chrom}-{rec['pos']}-{ref}-{alt}",
                "clinvar_id": rec["id"],
                "chrom": chrom,
                "pos": rec["pos"],
                "ref": ref,
                "alt": alt,
                "gene": genes[0],
                "clnsig": primary_significance(info.get("CLNSIG", "")),
                "review_stars": stars,
                "consequence": parse_molecular_consequence(info.get("MC")),
                "label": label,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("variant_id").reset_index(drop=True)
    return df


# Coarse groups used for the "by consequence type" analysis (Section III.F).
CONSEQUENCE_GROUPS = {
    "missense_variant": "missense",
    "synonymous_variant": "synonymous",
    "nonsense": "nonsense",
    "stop_gained": "nonsense",
    "stop_lost": "nonsense",
    "start_lost": "nonsense",
    "initiator_codon_variant": "nonsense",
    "splice_donor_variant": "splice",
    "splice_acceptor_variant": "splice",
    "splice_region_variant": "splice",
    "intron_variant": "intronic",
    "5_prime_UTR_variant": "utr",
    "3_prime_UTR_variant": "utr",
    "non-coding_transcript_variant": "noncoding",
    "non_coding_transcript_variant": "noncoding",
    "upstream_gene_variant": "noncoding",
    "downstream_gene_variant": "noncoding",
    "intergenic_variant": "noncoding",
}


def consequence_group(consequence: str) -> str:
    return CONSEQUENCE_GROUPS.get(consequence, "other")
