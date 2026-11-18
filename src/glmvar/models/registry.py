"""The three genomic language models compared in the study (Module C, Table 1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    tokenization: str
    finetune: str  # "lora" or "full"
    lora_targets: tuple[str, ...] = ()
    supports_mlm_zeroshot: bool = False
    max_length_bp: int = 1024
    extra: dict = field(default_factory=dict)


MODELS: dict[str, ModelSpec] = {
    "dnabert2": ModelSpec(
        key="dnabert2",
        hf_id="zhihan1996/DNABERT-2-117M",
        family="DNABERT-2 (Zhou et al., 2024)",
        tokenization="Byte-pair encoding",
        finetune="lora",
        lora_targets=("Wqkv",),
        supports_mlm_zeroshot=True,
    ),
    "nt_v2_250m": ModelSpec(
        key="nt_v2_250m",
        hf_id="InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
        family="Nucleotide Transformer v2 (Dalla-Torre et al., 2025)",
        tokenization="Non-overlapping 6-mers",
        finetune="lora",
        lora_targets=("query", "key", "value"),
        supports_mlm_zeroshot=True,
    ),
    "nt_v2_500m": ModelSpec(  # optional: only if GPU memory allows (Module C, Section III.D)
        key="nt_v2_500m",
        hf_id="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        family="Nucleotide Transformer v2 (Dalla-Torre et al., 2025)",
        tokenization="Non-overlapping 6-mers",
        finetune="lora",
        lora_targets=("query", "key", "value"),
        supports_mlm_zeroshot=True,
    ),
    "hyenadna_medium": ModelSpec(
        key="hyenadna_medium",
        hf_id="LongSafari/hyenadna-medium-160k-seqlen-hf",
        family="HyenaDNA (Nguyen et al., 2023)",
        tokenization="Single nucleotides",
        finetune="full",
        supports_mlm_zeroshot=False,
        max_length_bp=160_000,
    ),
}


def get_spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise KeyError(f"unknown model {key!r}; choose from {sorted(MODELS)}") from exc
