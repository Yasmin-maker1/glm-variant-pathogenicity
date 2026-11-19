"""Load a pretrained backbone + tokenizer and wrap it for one of two training modes.

Modes (Section III.D):
  * ``finetune`` - LoRA for transformer models, full fine-tuning for HyenaDNA
  * ``frozen``   - backbone weights fixed, only the classification head trains

Each of the three checkpoints ships its own ``trust_remote_code`` model, and
they do not all register the same Auto* classes, so loading tries AutoModel
first and falls back to the masked-LM / causal-LM class and its base model.
"""

from __future__ import annotations

import logging

from .classifier import PairClassifier, hidden_size_of
from .registry import ModelSpec

LOG = logging.getLogger(__name__)


def _extra_kwargs(spec: ModelSpec) -> dict:
    kw = {"trust_remote_code": True}
    if spec.key.startswith("dnabert2"):
        # Known DNABERT-2 workaround for newer transformers versions:
        # pass the standard BertConfig explicitly.
        from transformers.models.bert.configuration_bert import BertConfig

        kw["config"] = BertConfig.from_pretrained(spec.hf_id)
    return kw


def disable_dnabert2_triton() -> int:
    """Make DNABERT-2 use its plain PyTorch attention instead of its Triton kernel.

    DNABERT-2's optional flash-attention kernel was written for an old Triton
    API (``tl.dot(..., trans_b=True)``) and fails to compile with current
    Triton. Its code falls back to standard attention when the module-level
    ``flash_attn_qkvpacked_func`` is None, so we set that after loading.
    Returns the number of modules patched.
    """
    import sys

    patched = 0
    for name, mod in list(sys.modules.items()):
        if name.endswith("bert_layers") and getattr(mod, "flash_attn_qkvpacked_func", None) is not None:
            mod.flash_attn_qkvpacked_func = None
            patched += 1
    return patched


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)


def load_backbone(spec: ModelSpec):
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM

    kw = _extra_kwargs(spec)
    errors = []
    try:
        model = AutoModel.from_pretrained(spec.hf_id, **kw)
        if disable_dnabert2_triton():
            LOG.info("DNABERT-2: using PyTorch attention (Triton kernel disabled)")
        return model
    except Exception as exc:  # noqa: BLE001 - remote code raises many types
        errors.append(f"AutoModel: {exc}")
    for cls in (AutoModelForMaskedLM, AutoModelForCausalLM):
        try:
            model = cls.from_pretrained(spec.hf_id, **kw)
            disable_dnabert2_triton()
            LOG.info("loaded %s via %s; using its base model", spec.hf_id, cls.__name__)
            return model.base_model
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(f"could not load {spec.hf_id}:\n" + "\n".join(errors))


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
