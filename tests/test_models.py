"""Model-side tests that run offline with a tiny random encoder and a toy tokenizer."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from glmvar.models.classifier import PairClassifier  # noqa: E402
from glmvar.models.tokens import pool_mask, token_index_at, token_spans  # noqa: E402
from glmvar.train import train_model  # noqa: E402


def test_token_spans_bpe_and_kmers():
    bpe = ["[CLS]", "ACG", "T", "GGAC", "[SEP]"]
    assert token_spans(bpe) == [(0, 0), (0, 3), (3, 4), (4, 8), (8, 8)]
    assert token_index_at(bpe, 3) == 2 and token_index_at(bpe, 5) == 3
    kmers = ["<cls>", "ACGTAC", "GGTTAA"]
    assert pool_mask(kmers, center=6, radius=0) == [0, 0, 1]
    assert pool_mask(kmers, center=5, radius=1) == [0, 1, 1]


class CharTokenizer:
    """Minimal HF-like single-nucleotide tokenizer (HyenaDNA style)."""

    vocab = {"[PAD]": 0, "[CLS]": 1, "[MASK]": 2, "[UNK]": 3, "A": 4, "C": 5, "G": 6, "T": 7, "N": 8}
    inv = {v: k for k, v in vocab.items()}
    pad_token_id, mask_token_id, unk_token_id = 0, 2, 3

    def __call__(self, seqs, padding="longest", return_tensors="pt", add_special_tokens=True):
        ids = [[1] + [self.vocab.get(c, 3) for c in s] for s in seqs]
        L = max(map(len, ids))
        attn = [[1] * len(x) + [0] * (L - len(x)) for x in ids]
        ids = [x + [0] * (L - len(x)) for x in ids]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}

    def convert_ids_to_tokens(self, ids):
        return [self.inv[i] for i in ids]

    def convert_tokens_to_ids(self, tok):
        return self.vocab.get(tok, 3)


class TinyEncoder(torch.nn.Module):
    def __init__(self, vocab=9, hidden=16):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden})()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.conv = torch.nn.Conv1d(hidden, hidden, 5, padding=2)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids).transpose(1, 2)
        return (torch.relu(self.conv(h)).transpose(1, 2),)


def _toy_df(n=240, L=32, seed=0):
    """Label is 1 iff the alternate base is G or C: a learnable signal at the center."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        seq = "".join(rng.choice(list("ACGT"), L))
        alt = rng.choice([b for b in "ACGT" if b != seq[L // 2]])
        rows.append({"variant_id": f"v{i}", "ref_seq": seq, "alt_seq": seq[: L // 2] + alt + seq[L // 2 + 1 :],
                     "label": int(alt in "GC"), "split": ["train", "train", "train", "val", "test"][i % 5]})
    return pd.DataFrame(rows)


def test_pair_classifier_forward_backward():
    tok = CharTokenizer()
    model = PairClassifier(TinyEncoder(), 16)
    r = tok(["ACGTACGT", "TTTTAAAA"])
    a = tok(["ACGTCCGT", "TTTTGAAA"])
    pool = torch.ones_like(r["input_ids"], dtype=torch.float32)
    out = model(r["input_ids"], r["attention_mask"], pool, a["input_ids"], a["attention_mask"], pool)
    assert out.shape == (2,)
    out.sum().backward()
    assert model.backbone.emb.weight.grad is not None


def test_frozen_backbone_gets_no_grad():
    model = PairClassifier(TinyEncoder(), 16, freeze_backbone=True)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_training_learns_toy_signal(tmp_path):
    torch.manual_seed(0)
    cfg = {"training": {"batch_size": 16, "micro_batch_size": 8, "epochs": 15, "patience": 5,
                        "learning_rate": 3e-3, "head_learning_rate": 3e-3, "pool_radius": 2}}
    summary = train_model(PairClassifier(TinyEncoder(), 16), CharTokenizer(), _toy_df(), cfg, tmp_path, seed=0, device="cpu")
    assert summary["best_val_auroc"] > 0.8
    test = pd.read_csv(tmp_path / "predictions_test.csv")
    assert len(test) == 48 and test.score.between(0, 1).all()


def test_disable_dnabert2_triton_patches_loaded_module():
    import sys
    import types

    from glmvar.models.loading import disable_dnabert2_triton

    fake = types.ModuleType("transformers_modules.zhihan1996.DNABERT_2_117M.abc.bert_layers")
    fake.flash_attn_qkvpacked_func = lambda *a: None
    sys.modules[fake.__name__] = fake
    try:
        assert disable_dnabert2_triton() == 1
        assert fake.flash_attn_qkvpacked_func is None
        assert disable_dnabert2_triton() == 0
    finally:
        del sys.modules[fake.__name__]


class BpeLikeTokenizer(CharTokenizer):
    """Merges 'AA' into one token, so ref and alt can get different lengths."""

    vocab = {**CharTokenizer.vocab, "AA": 9}
    inv = {v: k for k, v in vocab.items()}

    def __call__(self, seqs, padding="longest", return_tensors="pt", add_special_tokens=True):
        ids = []
        for s in seqs:
            out, i = [1], 0
            while i < len(s):
                if s[i : i + 2] == "AA":
                    out.append(9); i += 2
                else:
                    out.append(self.vocab.get(s[i], 3)); i += 1
            ids.append(out)
        L = max(map(len, ids))
        attn = [[1] * len(x) + [0] * (L - len(x)) for x in ids]
        ids = [x + [0] * (L - len(x)) for x in ids]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn)}


def test_collator_pads_ref_and_alt_to_same_length():
    from glmvar.train import PairCollator

    batch = PairCollator(BpeLikeTokenizer(), pool_radius=1)([("CCAACC", "CCATCC", 1.0)])
    assert batch["ref_ids"].shape == batch["alt_ids"].shape
    model = PairClassifier(TinyEncoder(vocab=10), 16)
    out = model(**{k: v for k, v in batch.items() if k != "labels"})
    assert out.shape == (1,)
