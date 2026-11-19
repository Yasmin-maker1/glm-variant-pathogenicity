"""End-to-end checks of the command-line steps on tiny offline data."""


import numpy as np
import pandas as pd
import pytest
import yaml

from glmvar.cli import main


def _config(tmp_path, clinvar_vcf, genome_fasta, **data):
    fasta, _ = genome_fasta
    cfg = {
        "paths": {"raw": str(tmp_path), "processed": str(tmp_path / "proc"), "runs": str(tmp_path / "runs"), "reports": str(tmp_path / "rep")},
        "data": {"clinvar_vcf": str(clinvar_vcf), "reference_fasta": str(fasta), "min_review_stars": 1, "drop_multi_gene": True,
                 "window_bp": 64, "extra_windows_bp": [], "split_fractions": [0.5, 0.25, 0.25], "split_seed": 1, **data},
        "training": {}, "evaluation": {"bootstrap": 10},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_build_dataset_with_caps(tmp_path, clinvar_vcf, genome_fasta):
    cfg = _config(tmp_path, clinvar_vcf, genome_fasta, max_variants_per_split={"train": 1})
    main(["--config", str(cfg), "build-dataset"])
    df = pd.read_parquet(tmp_path / "proc" / "variants_w64.parquet")
    assert (df.split == "train").sum() <= 1
    assert set(df.columns) >= {"variant_id", "gene", "label", "ref_seq", "alt_seq", "split"}
    assert df.ref_seq.str.len().eq(64).all()
    assert df.groupby("gene").split.nunique().max() == 1


def test_cadd_tabix_lookup(tmp_path):
    pysam = pytest.importorskip("pysam")
    raw = tmp_path / "cadd.tsv"
    raw.write_text("## CADD GRCh38-v1.7\n#Chrom\tPos\tRef\tAlt\tRawScore\tPHRED\n"
                   "1\t100\tA\tG\t1.0\t12.5\n1\t100\tA\tT\t2.0\t22.0\n1\t200\tC\tT\t0.1\t3.1\n")
    pysam.tabix_compress(str(raw), str(raw) + ".gz", force=True)
    pysam.tabix_index(str(raw) + ".gz", seq_col=0, start_col=1, end_col=1, meta_char="#", force=True)
    from glmvar.cadd import lookup_tabix

    df = pd.DataFrame({"chrom": ["1", "1", "1"], "pos": [100, 200, 300], "ref": ["A", "C", "G"], "alt": ["T", "T", "A"]})
    s = lookup_tabix(df, str(raw) + ".gz", threads=2)
    assert s[0] == 22.0 and s[1] == 3.1 and np.isnan(s[2])


def test_lora_wrapped_backbone_still_gets_attention_mask():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from glmvar.models.classifier import backbone_hidden_states
    from glmvar.models.loading import apply_lora
    from glmvar.models.registry import ModelSpec

    cfg = transformers.BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    torch.manual_seed(0)
    bert = transformers.BertModel(cfg).eval()
    spec = ModelSpec("toy", "toy", "toy", "toy", "lora", lora_targets=("query", "key", "value"))
    wrapped = apply_lora(bert, spec)
    ids = torch.tensor([[1, 5, 6, 7, 0, 0]])
    full = torch.ones_like(ids)
    masked = torch.tensor([[1, 1, 1, 1, 0, 0]])
    with torch.no_grad():
        a = backbone_hidden_states(wrapped, ids, full)
        b = backbone_hidden_states(wrapped, ids, masked)
    assert not torch.allclose(a[:, :4], b[:, :4])  # mask was actually used


def test_train_cli_end_to_end_and_report(tmp_path, clinvar_vcf, genome_fasta, monkeypatch):
    pytest.importorskip("torch")
    from test_models import CharTokenizer, TinyEncoder

    import glmvar.models.loading as loading

    cfg = _config(tmp_path, clinvar_vcf, genome_fasta)
    main(["--config", str(cfg), "build-dataset"])
    monkeypatch.setattr(loading, "load_tokenizer", lambda spec: CharTokenizer())
    monkeypatch.setattr(loading, "load_backbone", lambda spec: TinyEncoder())
    args = ["--config", str(cfg), "train", "--model", "hyenadna_medium", "--mode", "finetune", "--seed", "0", "--skip-existing"]
    main(args)
    summary = tmp_path / "runs" / "hyenadna_medium_finetune" / "seed0" / "summary.json"
    assert summary.exists()
    mtime = summary.stat().st_mtime
    main(args)  # second call is skipped
    assert summary.stat().st_mtime == mtime
