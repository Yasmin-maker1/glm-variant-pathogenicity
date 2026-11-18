import numpy as np
import pandas as pd
import pytest

from glmvar.data.clinvar import consequence_group, label_from_clnsig, load_clinvar
from glmvar.data.sequences import ReferenceGenome, add_sequences, variant_window
from glmvar.data.split import check_no_leakage, gene_split, subsample_train_genes


def test_labels():
    assert label_from_clnsig("Pathogenic") == 1
    assert label_from_clnsig("Likely_pathogenic") == 1
    assert label_from_clnsig("Pathogenic|risk_factor") == 1
    assert label_from_clnsig("Benign/Likely_benign") == 0
    assert label_from_clnsig("Uncertain_significance") is None
    assert label_from_clnsig("Conflicting_classifications_of_pathogenicity") is None


def test_load_clinvar_filters(clinvar_vcf):
    df = load_clinvar(clinvar_vcf, min_stars=1, drop_multi_gene=True)
    # kept: 100 (P), 200 (B), 300 (LP), 700 (LB); dropped: VUS, 0-star, multi-gene, indel
    assert sorted(df.pos) == [100, 200, 300, 700]
    assert df.set_index("pos").label.to_dict() == {100: 1, 200: 0, 300: 1, 700: 0}
    assert df.set_index("pos").loc[100, "consequence"] == "missense_variant"
    assert consequence_group("stop_gained") == "nonsense"


def test_window_centered_and_alt_applied(genome_fasta):
    path, seq = genome_fasta
    g = ReferenceGenome(path)
    ref, alt, c = variant_window(g, "1", 101, seq[100], "N" if seq[100] != "N" else "A", 64)
    assert len(ref) == len(alt) == 64 and c == 32
    assert ref[c] == seq[100]
    assert ref[:c] == alt[:c] and ref[c + 1 :] == alt[c + 1 :]
    assert ref == seq[100 - 32 : 100 + 32]


def test_window_pads_at_chromosome_edge(genome_fasta):
    path, seq = genome_fasta
    g = ReferenceGenome(path)
    ref, _, c = variant_window(g, "chr1", 3, seq[2], "A", 16)
    assert ref.startswith("N" * 6) and ref[c] == seq[2]


def test_ref_mismatch_dropped(genome_fasta):
    path, seq = genome_fasta
    bad = "A" if seq[9] != "A" else "C"
    df = pd.DataFrame({"chrom": ["1", "1"], "pos": [10, 20], "ref": [bad, seq[19]], "alt": ["G", "G"]})
    out = add_sequences(df, ReferenceGenome(path), 32)
    assert len(out) == 1 and out.attrs["n_dropped_ref_mismatch"] == 1


def test_gene_split_no_leakage_and_fractions():
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(500)]
    df = pd.DataFrame({"gene": rng.choice(genes, size=20000), "label": rng.integers(0, 2, 20000)})
    out = gene_split(df, (0.8, 0.1, 0.1), seed=1)
    check_no_leakage(out)
    frac = out.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.8) < 0.03 and abs(frac["test"] - 0.1) < 0.03
    # deterministic
    assert out.split.equals(gene_split(df, (0.8, 0.1, 0.1), seed=1).split)


def test_subsample_keeps_eval_sets():
    df = pd.DataFrame({"gene": [f"G{i}" for i in range(100)], "split": ["train"] * 80 + ["val"] * 10 + ["test"] * 10})
    sub = subsample_train_genes(df, 0.25, seed=0)
    assert (sub.split == "train").sum() == 20
    assert (sub.split != "train").sum() == 20
    with pytest.raises(ValueError):
        subsample_train_genes(df, 0)
