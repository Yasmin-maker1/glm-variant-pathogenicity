"""Command-line entry point: ``glmvar <command> [options]``.

  glmvar build-dataset  --config configs/default.yaml
  glmvar train          --config configs/default.yaml --model dnabert2 --mode finetune --seed 0
  glmvar zeroshot       --config configs/default.yaml --model nt_v2_250m
  glmvar cadd           --config configs/default.yaml
  glmvar report         --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .utils import load_config, save_json, setup_logging

LOG = logging.getLogger("glmvar")


def _variants_path(cfg, window: int | None = None) -> Path:
    w = window or cfg["data"]["window_bp"]
    return Path(cfg["paths"]["processed"]) / f"variants_w{w}.parquet"


def cmd_build_dataset(cfg, args):
    from .data.clinvar import load_clinvar
    from .data.sequences import ReferenceGenome, add_sequences
    from .data.split import check_no_leakage, gene_split

    d = cfg["data"]
    LOG.info("parsing ClinVar %s", d["clinvar_vcf"])
    df = load_clinvar(d["clinvar_vcf"], min_stars=d.get("min_review_stars", 1), drop_multi_gene=d.get("drop_multi_gene", True))
    LOG.info("%d labeled SNVs (%d pathogenic, %d benign)", len(df), int(df.label.sum()), int((df.label == 0).sum()))
    if d.get("vep_tsv"):
        from .data.vep import merge_vep, read_vep_tab

        df = merge_vep(df, read_vep_tab(d["vep_tsv"]))
    df = gene_split(df, tuple(d["split_fractions"]), seed=d["split_seed"])
    check_no_leakage(df)
    genome = ReferenceGenome(d["reference_fasta"])
    windows = sorted({d["window_bp"], *d.get("extra_windows_bp", [])}) if args.all_windows else [d["window_bp"]]
    stats = {}
    for w in windows:
        out = add_sequences(df, genome, w)
        path = _variants_path(cfg, w)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        stats[w] = {"n": len(out), "dropped_ref_mismatch": out.attrs.get("n_dropped_ref_mismatch", 0),
                    "by_split": out.groupby("split").label.agg(["size", "sum"]).to_dict()}
        LOG.info("window %d bp -> %s (%d variants)", w, path, len(out))
    save_json(stats, Path(cfg["paths"]["processed"]) / "dataset_stats.json")


def cmd_train(cfg, args):
    from .data.split import subsample_train_genes
    from .models.loading import build_classifier, count_trainable, load_tokenizer
    from .models.registry import get_spec
    from .train import train_model

    spec = get_spec(args.model)
    df = pd.read_parquet(_variants_path(cfg, args.window))
    if args.train_fraction < 1:
        df = subsample_train_genes(df, args.train_fraction, seed=args.seed)
    if args.limit:
        df = df.groupby("split", group_keys=False).apply(lambda g: g.sample(min(len(g), args.limit), random_state=0))
    if args.learning_rate:
        cfg["training"]["learning_rate"] = args.learning_rate
    tok = load_tokenizer(spec)
    model = build_classifier(spec, args.mode, lora_cfg=cfg.get("lora"), head_cfg=cfg.get("head"))
    trainable, total = count_trainable(model)
    LOG.info("%s/%s: %.2fM trainable of %.1fM parameters", spec.key, args.mode, trainable / 1e6, total / 1e6)
    name = args.run_name or f"{spec.key}_{args.mode}" + (f"_w{args.window}" if args.window else "") + (f"_frac{args.train_fraction:g}" if args.train_fraction < 1 else "")
    out = Path(cfg["paths"]["runs"]) / name / f"seed{args.seed}"
    summary = train_model(model, tok, df, cfg, out, seed=args.seed)
    summary.update({"model": spec.key, "hf_id": spec.hf_id, "mode": args.mode, "trainable_params": trainable,
                    "total_params": total, "train_fraction": args.train_fraction, "window_bp": args.window or cfg["data"]["window_bp"]})
    save_json(summary, out / "summary.json")


def cmd_zeroshot(cfg, args):
    from .models.registry import get_spec
    from .zeroshot import llr_scores, load_mlm

    spec = get_spec(args.model)
    if not spec.supports_mlm_zeroshot:
        raise SystemExit(f"{spec.key} is not a masked language model; zero-shot LLR is not defined")
    df = pd.read_parquet(_variants_path(cfg))
    df = df[df.split.isin(["val", "test"])].reset_index(drop=True)
    model, tok = load_mlm(spec.hf_id)
    df["score"] = llr_scores(model, tok, df, batch_size=args.batch_size)
    LOG.info("zero-shot coverage: %.1f%%", 100 * df.score.notna().mean())
    out = Path(cfg["paths"]["runs"]) / f"{spec.key}_zeroshot" / "seed0"
    out.mkdir(parents=True, exist_ok=True)
    for s in ("val", "test"):
        df[df.split == s][["variant_id", "label", "score", "split"]].to_csv(out / f"predictions_{s}.csv", index=False)


def cmd_cadd(cfg, args):
    from .cadd import lookup_table, lookup_tabix

    c = cfg["cadd"]
    df = pd.read_parquet(_variants_path(cfg))
    df = df[df.split.isin(["val", "test"])].reset_index(drop=True)
    if c.get("scored_table"):
        df["score"] = lookup_table(df, c["scored_table"], c.get("column", "PHRED"))
    else:
        df["score"] = lookup_tabix(df, c["whole_genome_tsv"], c.get("column", "PHRED"))
    LOG.info("CADD coverage: %.1f%%", 100 * df.score.notna().mean())
    out = Path(cfg["paths"]["runs"]) / "cadd" / "seed0"
    out.mkdir(parents=True, exist_ok=True)
    for s in ("val", "test"):
        df[df.split == s][["variant_id", "label", "score", "split"]].to_csv(out / f"predictions_{s}.csv", index=False)


def cmd_export_vcf(cfg, args):
    from .data.vep import write_vcf

    df = pd.read_parquet(_variants_path(cfg))
    LOG.info("wrote %s", write_vcf(df, Path(cfg["paths"]["processed"]) / "variants.vcf"))


def cmd_report(cfg, args):
    from .evaluate import write_report
    from .plots import plot_all

    tables = write_report(cfg["paths"]["runs"], _variants_path(cfg), cfg["paths"]["reports"], n_boot=cfg["evaluation"]["bootstrap"])
    plot_all(cfg["paths"]["runs"], cfg["paths"]["reports"])
    print(tables["summary"].to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="glmvar", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-dataset")
    b.add_argument("--all-windows", action="store_true", help="also build the long windows for HyenaDNA")
    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--mode", choices=["finetune", "frozen"], default="finetune")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--learning-rate", type=float)
    t.add_argument("--train-fraction", type=float, default=1.0)
    t.add_argument("--window", type=int, help="window length in bp (default: data.window_bp)")
    t.add_argument("--limit", type=int, help="debug: cap variants per split")
    t.add_argument("--run-name")
    z = sub.add_parser("zeroshot")
    z.add_argument("--model", required=True)
    z.add_argument("--batch-size", type=int, default=16)
    sub.add_parser("cadd")
    sub.add_parser("export-vcf")
    sub.add_parser("report")

    args = p.parse_args(argv)
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    {"build-dataset": cmd_build_dataset, "train": cmd_train, "zeroshot": cmd_zeroshot, "cadd": cmd_cadd,
     "export-vcf": cmd_export_vcf, "report": cmd_report}[args.command](cfg, args)


if __name__ == "__main__":
    main()
