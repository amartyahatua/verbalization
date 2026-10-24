"""
Minimal mock model/tokenizer/SAE/data for offline pipeline validation
(no internet access or real weights required).

NOT a real language model — its only purpose is to check that
build_direction_bank.py's control flow, tensor shapes, and hook logic
are correct before running it against real weights and real data.
Swap `--dry_run` off and edit config.py to point at real models.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict


class MockTokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors="pt"):
        seq_len = min(32, max(4, len(text) // 4))
        ids = torch.randint(0, self.vocab_size, (1, seq_len))
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(text)) % self.vocab_size]


class MockLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return (self.proj(x),)


class MockInner(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(d_model) for _ in range(n_layers)])


class _Output:
    logits: torch.Tensor


class MockModel(nn.Module):
    """Mimics the HF interface this pipeline relies on: .model.layers[i], forward hooks, .logits"""

    def __init__(self, vocab_size: int = 1000, d_model: int = 64, n_layers: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.model = MockInner(d_model, n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            (h,) = layer(h)
        logits = self.lm_head(h)
        out = _Output()
        out.logits = logits
        return out

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class MockSAE:
    def __init__(self, d_model: int, d_sae: int):
        class _Cfg:
            pass
        self.cfg = _Cfg()
        self.cfg.d_sae = d_sae
        self.W_dec = torch.randn(d_sae, d_model)


def build_mock_model_and_sae(layers: List[int], device: str) -> Tuple[MockModel, MockTokenizer, Dict[int, MockSAE]]:
    d_model = 64
    model = MockModel(vocab_size=1000, d_model=d_model, n_layers=max(layers) + 1).to(device)
    model.eval()
    tokenizer = MockTokenizer(vocab_size=1000)
    sae_by_layer = {l: MockSAE(d_model, d_sae=512) for l in layers}
    return model, tokenizer, sae_by_layer


def mock_contexts_and_counterfact(n_contexts: int = 20, n_facts: int = 10):
    contexts = [
        f"This is example context number {i} about something reasonably interesting."
        for i in range(n_contexts)
    ]
    counterfact = [
        {
            "subject": f"Entity{i}",
            "relation": "has property",
            "object_true": f"TrueVal{i}",
            "object_new": f"FalseVal{i}",
        }
        for i in range(n_facts)
    ]
    return contexts, counterfact


class MockGenerativeTokenizer:
    """
    Like MockTokenizer, but also supports .decode() and .eos_token_id
    so it can drive real .generate() calls in tests -- used for Stage
    2 (and any later stage that needs generation) rather than
    Stage 1's plain MockTokenizer, which only needs single forward
    passes. Meant to be paired with a REAL (tiny, randomly
    initialized) HF model class in tests, not the custom MockModel
    above, since MockModel has no .generate() implementation.

    IMPORTANT: deterministic by design -- the same text always
    produces the same token ids (seeded from hash(text)), the same
    way a real tokenizer would. An earlier version used un-seeded
    torch.randint() here, which silently made every call to the SAME
    text return DIFFERENT ids -- harmless for Stages 1-3's dry runs
    (which only check that the pipeline runs, not that anything is
    learned), but it meant Stage 4's training convergence tests were
    training against a fresh random target every single step, with
    nothing consistent to learn. Caught via a real convergence check,
    not by inspection -- worth remembering when trusting a mock's
    "it ran without crashing" as sufficient.
    """

    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1

    def __call__(self, text, return_tensors="pt"):
        seq_len = min(16, max(4, len(text) // 4))
        gen = torch.Generator().manual_seed(abs(hash(text)) % (2**31))
        ids = torch.randint(0, self.vocab_size - 1, (1, seq_len), generator=gen)
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return [abs(hash(text)) % (self.vocab_size - 1)]

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if skip_special_tokens:
            ids = [i for i in ids if i != self.eos_token_id]
        return " ".join(f"tok{i}" for i in ids)
