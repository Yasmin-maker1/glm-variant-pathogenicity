"""Turn each variant into a reference and an alternate DNA window (Section III.B)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReferenceGenome:
    """Thin wrapper over an indexed FASTA. Uses pysam when available.

    Falls back to loading the FASTA into memory, which is only sensible for the
    small test fixtures. Chromosome names with or without a 'chr' prefix work.
    """

    def __init__(self, fasta_path: str | Path):
        self.path = str(fasta_path)
        try:
            import pysam

            self._fa = pysam.FastaFile(self.path)
            self._names = set(self._fa.references)
            self._mem = None
        except ImportError:
            self._fa = None
            self._mem = self._load(self.path)
            self._names = set(self._mem)

    @staticmethod
    def _load(path: str) -> dict[str, str]:
        seqs: dict[str, list[str]] = {}
        name = None
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    name = line[1:].split()[0]
                    seqs[name] = []
                elif name:
                    seqs[name].append(line)
        return {k: "".join(v) for k, v in seqs.items()}

    def _resolve(self, chrom: str) -> str:
        for cand in (chrom, f"chr{chrom}", chrom.removeprefix("chr"), "chrM" if chrom == "MT" else chrom):
            if cand in self._names:
                return cand
        raise KeyError(f"chromosome {chrom!r} not in {self.path}")

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        """0-based, half-open. Pads with 'N' beyond chromosome ends."""
        name = self._resolve(chrom)
        length = self._fa.get_reference_length(name) if self._fa else len(self._mem[name])
        left_pad = max(0, -start0)
        right_pad = max(0, end0 - length)
        s, e = max(0, start0), min(length, end0)
        seq = self._fa.fetch(name, s, e) if self._fa else self._mem[name][s:e]
        return ("N" * left_pad + seq + "N" * right_pad).upper()


def variant_window(genome: ReferenceGenome, chrom: str, pos: int, ref: str, alt: str, length: int) -> tuple[str, str, int]:
    """Return (ref_seq, alt_seq, center_index) for a 1-based VCF position.

    The variant base sits at index ``length // 2`` of the window.
    """
    center = length // 2
    start0 = (pos - 1) - center
    ref_seq = genome.fetch(chrom, start0, start0 + length)
    if ref_seq[center] != ref:
        raise ValueError(f"reference mismatch at {chrom}:{pos}: genome has {ref_seq[center]}, VCF has {ref}")
    alt_seq = ref_seq[:center] + alt + ref_seq[center + 1 :]
    return ref_seq, alt_seq, center


def add_sequences(df: pd.DataFrame, genome: ReferenceGenome, length: int) -> pd.DataFrame:
    """Add ref_seq/alt_seq columns; drops (and reports) variants whose REF disagrees with the genome."""
    ref_seqs, alt_seqs, keep = [], [], []
    for r in df.itertuples():
        try:
            rs, as_, _ = variant_window(genome, r.chrom, r.pos, r.ref, r.alt, length)
        except (ValueError, KeyError):
            keep.append(False)
            continue
        keep.append(True)
        ref_seqs.append(rs)
        alt_seqs.append(as_)
    out = df.loc[keep].copy()
    out["ref_seq"] = ref_seqs
    out["alt_seq"] = alt_seqs
    out.attrs["n_dropped_ref_mismatch"] = int(len(df) - len(out))
    return out.reset_index(drop=True)
