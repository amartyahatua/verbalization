"""
SAE feature direction extraction via SAELens.

Loads a pretrained SAE (e.g. Llama-Scope, Gemma-Scope) per layer and
returns a stratified sample of decoder-column directions, tagged with
a rough "abstraction" proxy (activation frequency: rarer features
tend to be more specific/concrete, common ones more abstract) so the
sample spans the abstraction range called for in Section 5.1.

`sae_lens` is imported lazily so this module stays importable in the
dry-run test without the package installed.
"""

import numpy as np
import torch
from typing import List, Dict, Optional


def load_sae(release: str, layer: int, device: str = "cuda", hook_point: str = "hook_resid_pre"):
    """
    `hook_point` varies by release — check the release's valid-IDs list
    (SAELens will print it in the error message if you get it wrong).
    E.g. "gpt2-small-res-jb" uses "hook_resid_pre"; some other releases
    (Llama-Scope, the Qwen2-0.5B SAE) use "hook_resid_post"; Gemma Scope
    uses a different id scheme entirely ("layer_{L}/width_{W}/canonical").
    """
    from sae_lens import SAE

    sae_id = f"blocks.{layer}.{hook_point}"
    sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae


def stratified_feature_sample(
    sae,
    n_features: int,
    activation_frequencies: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Sample feature indices spanning the abstraction range.

    If per-feature activation frequencies are available (e.g. from a
    precomputed Neuronpedia/SAELens stats file), stratify by
    frequency decile. Otherwise fall back to a uniform random sample
    (and you should note in your write-up that stratification wasn't
    applied for that layer).
    """
    n_total = sae.cfg.d_sae
    n_features = min(n_features, n_total)

    if activation_frequencies is None:
        idx = np.random.choice(n_total, size=n_features, replace=False)
        return sorted(idx.tolist())

    order = np.argsort(activation_frequencies)
    bins = np.array_split(order, 10)
    per_bin = max(1, n_features // 10)
    sampled = []
    for b in bins:
        take = min(per_bin, len(b))
        sampled.extend(np.random.choice(b, size=take, replace=False).tolist())
    return sorted(sampled[:n_features])


def get_sae_directions(sae, feature_indices: List[int]) -> Dict[int, torch.Tensor]:
    """Return {feature_idx: decoder_column_vector} for the given features."""
    W_dec = sae.W_dec.detach().cpu()  # [d_sae, d_model]
    return {i: W_dec[i].clone() for i in feature_indices}
