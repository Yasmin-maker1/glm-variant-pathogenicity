"""Load a pretrained backbone + tokenizer and wrap it for one of three training modes.

Modes (Section III.D):
  * ``finetune`` - LoRA for transformer models, full fine-tuning for HyenaDNA
  * ``frozen``   - backbone weights fixed, only the classification head trains
"""

from __future__ import annotations

from .classifier import PairClassifier, hidden_size_of
from .registry import ModelSpec


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)


def load_backbone(spec: ModelSpec):
    from transformers import AutoModel

    return AutoModel.from_pretrained(spec.hf_id, trust_remote_code=True)


def apply_lora(backbone, spec: ModelSpec, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(spec.lora_targets), bias="none")
    return get_peft_model(backbone, cfg)


def build_classifier(spec: ModelSpec, mode: str, lora_cfg: dict | None = None, head_cfg: dict | None = None, backbone=None):
    """Return a PairClassifier ready to train. ``backbone`` can be injected for tests."""
    lora_cfg = lora_cfg or {}
    head_cfg = head_cfg or {}
    backbone = backbone if backbone is not None else load_backbone(spec)
    hidden = hidden_size_of(backbone)
    if mode == "frozen":
        return PairClassifier(backbone, hidden, freeze_backbone=True, **head_cfg)
    if mode != "finetune":
        raise ValueError(f"unknown mode {mode!r}")
    if spec.finetune == "lora":
        backbone = apply_lora(backbone, spec, **lora_cfg)
    return PairClassifier(backbone, hidden, freeze_backbone=False, **head_cfg)


def count_trainable(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
