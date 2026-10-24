"""
Probe direction extraction: mean-difference and logistic-regression
probes for known attributes, fit on CounterFact-style contrastive
data.

Section 5.1 deliberately calls for some probe directions that are
linearly decodable but causally inert ("present but unused"). This
module just fits the probes; whether a given probe turns out to be
causally inert is determined later, in causal_efficacy.py, by
steering with it and checking whether behavior changes.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Tuple

from model_utils import get_transformer_layers


def get_residual_activations(
    model, tokenizer, texts: List[str], layer: int, device: str
) -> torch.Tensor:
    """Return the final-token residual-stream activation at `layer` for each text."""
    acts = []
    captured = {}

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach().float()  # cast off bfloat16 here — sklearn/numpy can't handle it downstream
        return out

    handle = get_transformer_layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs)
                acts.append(captured["h"][0, -1].cpu())
    finally:
        handle.remove()
    return torch.stack(acts)


def fit_mean_diff_probe(acts_pos: torch.Tensor, acts_neg: torch.Tensor) -> torch.Tensor:
    """Direction = normalized (mean(pos) - mean(neg))."""
    d = acts_pos.mean(0) - acts_neg.mean(0)
    return d / d.norm()


def fit_logistic_probe(acts_pos: torch.Tensor, acts_neg: torch.Tensor) -> torch.Tensor:
    """Direction = normalized logistic-regression weight vector."""
    X = torch.cat([acts_pos, acts_neg]).numpy()
    y = np.array([1] * len(acts_pos) + [0] * len(acts_neg))
    clf = LogisticRegression(max_iter=2000).fit(X, y)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    return w / w.norm()


def build_probe_directions(
    model,
    tokenizer,
    counterfact: List[Dict],
    layers: List[int],
    device: str,
    n_attributes: int = 20,
) -> Dict[Tuple[int, str], torch.Tensor]:
    """
    Group CounterFact records by relation, build pos/neg text pairs
    (true-object vs. counterfactual-object completions of the same
    template), and fit both probe types per (layer, relation).
    """
    by_relation: Dict[str, List[Dict]] = {}
    for r in counterfact:
        by_relation.setdefault(r["relation"], []).append(r)

    directions = {}
    for relation, records in list(by_relation.items())[:n_attributes]:
        pos_texts = [f"{r['subject']} {relation} {r['object_true']}" for r in records]
        neg_texts = [f"{r['subject']} {relation} {r['object_new']}" for r in records]

        for layer in layers:
            acts_pos = get_residual_activations(model, tokenizer, pos_texts, layer, device)
            acts_neg = get_residual_activations(model, tokenizer, neg_texts, layer, device)
            directions[(layer, f"{relation}::meandiff")] = fit_mean_diff_probe(acts_pos, acts_neg)
            directions[(layer, f"{relation}::logreg")] = fit_logistic_probe(acts_pos, acts_neg)
    return directions
