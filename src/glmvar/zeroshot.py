"""Zero-shot masked-language-model scores (Section III.D, second reference point).

For DNABERT-2 and the Nucleotide Transformer we mask the token that contains
the variant base and compare the model's log-probabilities of the reference
token and of the same token with the alternate base:

    LLR = log p(alt_token) - log p(ref_token)

A strongly negative LLR means the model finds the variant unlikely, which we
treat as evidence of pathogenicity, so the score reported is ``-LLR``.
If the alternate token string is not in the vocabulary (possible with BPE),
the score is NaN and the coverage is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .models.tokens import token_index_at, token_spans


def load_mlm(hf_id: str):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    kw = {"trust_remote_code": True}
    if "DNABERT-2" in hf_id:
        from transformers.models.bert.configuration_bert import BertConfig

        kw["config"] = BertConfig.from_pretrained(hf_id)
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(hf_id, **kw)
    from .models.loading import disable_dnabert2_triton

    disable_dnabert2_triton()
    return model, tok


@torch.no_grad()
def llr_scores(model, tokenizer, df: pd.DataFrame, batch_size: int = 16, device: str | None = None) -> np.ndarray:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()
    unk = tokenizer.unk_token_id
    scores = np.full(len(df), np.nan, dtype=np.float64)
    rows = list(df.itertuples(index=False))
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        enc = tokenizer([r.ref_seq for r in chunk], return_tensors="pt", padding="longest")
        ids = enc["input_ids"].clone()
        targets = []  # (row_in_batch, token_idx, ref_id, alt_id)
        for b, r in enumerate(chunk):
            tokens = tokenizer.convert_ids_to_tokens(ids[b].tolist())
            center = len(r.ref_seq) // 2
            ti = token_index_at(tokens, center)
            if ti is None:
                continue
            s, _ = token_spans(tokens)[ti]
            ref_tok = tokens[ti]
            offset = center - s
            alt_tok = ref_tok[:offset] + r.alt + ref_tok[offset + 1 :]
            alt_id = tokenizer.convert_tokens_to_ids(alt_tok)
            if alt_id is None or alt_id == unk:
                continue
            targets.append((b, ti, int(ids[b, ti]), int(alt_id)))
            ids[b, ti] = tokenizer.mask_token_id
        if not targets:
            continue
        kwargs = {"input_ids": ids.to(device)}
        if "attention_mask" in enc:
            kwargs["attention_mask"] = enc["attention_mask"].to(device)
        out = model(**kwargs)
        logits = out.logits if hasattr(out, "logits") else out[0]
        if logits.dim() != 3 or logits.shape[1] != ids.shape[1]:
            raise RuntimeError(f"unexpected MLM logits shape {tuple(logits.shape)}; zero-shot not supported for this model")
        logp = torch.log_softmax(logits.float(), dim=-1)
        for b, ti, ref_id, alt_id in targets:
            llr = (logp[b, ti, alt_id] - logp[b, ti, ref_id]).item()
            scores[start + b] = -llr
    return scores
