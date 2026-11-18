"""Map tokens back to base-pair positions so we can pool around the variant.

DNABERT-2 uses BPE (variable-length tokens), the Nucleotide Transformer uses
6-mers and HyenaDNA uses single nucleotides. Rather than special-casing each
tokenizer, we read every token's string and count how many bases it covers.
Special tokens ([CLS], <eos>, padding...) cover zero bases.
"""

from __future__ import annotations

DNA_CHARS = set("ACGTN")


def token_base_lengths(tokens: list[str]) -> list[int]:
    lengths = []
    for tok in tokens:
        t = tok.lstrip("#▁Ġ").upper()  # strip common subword markers
        lengths.append(len(t) if t and set(t) <= DNA_CHARS else 0)
    return lengths


def token_spans(tokens: list[str]) -> list[tuple[int, int]]:
    """(start, end) base-pair span of every token, half-open; (p, p) for special tokens."""
    spans, pos = [], 0
    for n in token_base_lengths(tokens):
        spans.append((pos, pos + n))
        pos += n
    return spans


def pool_mask(tokens: list[str], center: int, radius: int) -> list[int]:
    """1 for tokens overlapping [center - radius, center + radius], else 0.

    Falls back to all DNA-bearing tokens if none overlap (e.g. truncation).
    """
    lo, hi = center - radius, center + radius + 1
    spans = token_spans(tokens)
    mask = [1 if (e > s and s < hi and e > lo) else 0 for s, e in spans]
    if not any(mask):
        mask = [1 if e > s else 0 for s, e in spans]
    return mask


def token_index_at(tokens: list[str], base_pos: int) -> int | None:
    """Index of the token that contains ``base_pos``, or None."""
    for i, (s, e) in enumerate(token_spans(tokens)):
        if s <= base_pos < e:
            return i
    return None
