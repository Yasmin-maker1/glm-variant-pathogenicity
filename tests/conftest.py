import random

import pytest

CHROM_LEN = 5000


@pytest.fixture
def genome_fasta(tmp_path):
    rng = random.Random(0)
    seq = "".join(rng.choice("ACGT") for _ in range(CHROM_LEN))
    path = tmp_path / "ref.fa"
    path.write_text(">1\n" + "\n".join(seq[i : i + 60] for i in range(0, len(seq), 60)) + "\n")
    return path, seq


@pytest.fixture
def clinvar_vcf(tmp_path, genome_fasta):
    _, seq = genome_fasta
    rows = [
        # pos, ref-from-genome?, alt, CLNSIG, CLNREVSTAT, GENEINFO, MC
        (100, "G", "Pathogenic", "criteria_provided,_single_submitter", "GENEA:1", "SO:0001583|missense_variant"),
        (200, "T", "Benign", "reviewed_by_expert_panel", "GENEA:1", "SO:0001819|synonymous_variant"),
        (300, "C", "Likely_pathogenic", "criteria_provided,_multiple_submitters,_no_conflicts", "GENEB:2", "SO:0001587|nonsense"),
        (400, "A", "Uncertain_significance", "criteria_provided,_single_submitter", "GENEB:2", "SO:0001583|missense_variant"),
        (500, "G", "Benign/Likely_benign", "no_assertion_criteria_provided", "GENEC:3", "SO:0001627|intron_variant"),
        (600, "T", "Pathogenic|risk_factor", "criteria_provided,_single_submitter", "GENEC:3|GENED:4", "SO:0001583|missense_variant"),
        (700, "C", "Likely_benign", "criteria_provided,_single_submitter", "GENED:4", "SO:0001627|intron_variant"),
    ]
    lines = ["##fileformat=VCFv4.1", "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
    for i, (pos, alt, sig, rev, gene, mc) in enumerate(rows):
        ref = seq[pos - 1]
        if alt == ref:
            alt = {"A": "C", "C": "G", "G": "T", "T": "A"}[ref]
        info = f"CLNSIG={sig};CLNREVSTAT={rev};GENEINFO={gene};MC={mc};CLNVC=single_nucleotide_variant"
        lines.append(f"1\t{pos}\t{i}\t{ref}\t{alt}\t.\t.\t{info}")
    # an indel that must be dropped
    lines.append(f"1\t800\t99\t{seq[799]}A\t{seq[799]}\t.\t.\tCLNSIG=Pathogenic;CLNREVSTAT=reviewed_by_expert_panel;GENEINFO=GENEE:5")
    path = tmp_path / "clinvar.vcf"
    path.write_text("\n".join(lines) + "\n")
    return path
