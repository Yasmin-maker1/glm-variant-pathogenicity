"""Reference/alternate pair classifier shared by all three models (Section III.D).

Both sequences go through the same backbone. Each is reduced to one vector by
averaging token embeddings near the variant; [ref, alt, alt - ref] is fed to a
small feed-forward head that outputs a pathogenicity logit.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn


def backbone_hidden_states(backbone: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Return [batch, tokens, hidden] for HF models with differing output types."""
    kwargs = {"input_ids": input_ids}
    params = inspect.signature(backbone.forward).parameters
    if attention_mask is not None and "attention_mask" in params:
        kwargs["attention_mask"] = attention_mask
    if "output_hidden_states" in params:
        kwargs["output_hidden_states"] = True
    out = backbone(**kwargs)
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        return out.hidden_states[-1]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class PairClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, head_hidden: int = 256, dropout: float = 0.1, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_size, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def embed(self, input_ids, attention_mask, pool_mask):
        if self.freeze_backbone:
            with torch.no_grad():
                h = backbone_hidden_states(self.backbone, input_ids, attention_mask)
        else:
            h = backbone_hidden_states(self.backbone, input_ids, attention_mask)
        return masked_mean(h.float(), pool_mask)

    def forward(self, ref_ids, ref_attn, ref_pool, alt_ids, alt_attn, alt_pool) -> torch.Tensor:
        # One backbone pass over the stacked ref+alt batch.
        b = ref_ids.shape[0]
        emb = self.embed(
            torch.cat([ref_ids, alt_ids]),
            torch.cat([ref_attn, alt_attn]),
            torch.cat([ref_pool, alt_pool]),
        )
        r, a = emb[:b], emb[b:]
        return self.head(torch.cat([r, a, a - r], dim=-1)).squeeze(-1)


def hidden_size_of(backbone: nn.Module) -> int:
    cfg = getattr(backbone, "config", None)
    for attr in ("hidden_size", "d_model", "n_embd"):
        if cfg is not None and getattr(cfg, attr, None):
            return int(getattr(cfg, attr))
    raise ValueError("could not infer hidden size; pass it explicitly")
