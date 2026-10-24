"""
SAE feature direction extraction via SAELens.

Loads a pretrained SAE (e.g. Llama-Scope, Gemma-Scope) per layer and
returns a stratified sample of decoder-column directions, tagged with
a rough "abstraction" proxy (activation frequency: rarer features
tend to be more specific/concrete, common ones more abstract) so the
sample spans the abstraction range called for in Section 5.1.

`sae_lens` is imported lazily so this module stays importable in the
dry-run test without the package installed.

On sae_id schemes: every SAELens release picks its own `sae_id`
string format, and there is no way to guess it reliably -- it is NOT
always "blocks.{layer}.{hook_point}". Confirmed by inspecting a live
sae_lens 6.49.1 install (sae_lens.loading.pretrained_saes_directory):
  - "gpt2-small-res-jb":               "blocks.{layer}.hook_resid_pre"
  - "llama_scope_lxr_32x":             "l{layer}r_32x"   (NOT
    "blocks.{layer}.hook_resid_post" -- this is what silently broke
    the original version of this function: SAE.from_pretrained() just
    raises ValueError, it doesn't fall back or warn about the format)
  - "llama-3.1-8b-instruct-andyrdt":    "resid_post_layer_{layer}_trainer_1"
  - Gemma Scope releases:               "layer_{layer}/width_{W}/canonical"
Always verify a release's actual scheme before trusting a hardcoded
template -- if this raises, run:
    python -c "
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
    d = get_pretrained_saes_directory()
    print(d['<release>'].saes_map)
    "
and pass the matching template as CFG.sae_id_template.
"""

import numpy as np
import torch
from typing import List, Dict, Optional


def load_sae(release: str, layer: int, device: str = "cuda",
             sae_id_template: str = "blocks.{layer}.hook_resid_post"):
    """
    `sae_id_template` is a format string with a single `{layer}`
    placeholder, e.g. "l{layer}r_32x" for llama_scope releases or
    "resid_post_layer_{layer}_trainer_1" for andyrdt's Llama-3.1
    Instruct SAEs. This varies by release -- see the module docstring
    for how to look up the correct one, and don't assume the default
    here is right for whatever release you're pointing at.
    """
    from sae_lens import SAE

    sae_id = sae_id_template.format(layer=layer)
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
